"""Scan constant-velocity time-shift scores on development events.

This is a sensitivity diagnostic using the frozen 4200 m/s model. It does not
claim that a different velocity is an independently retrained optimum.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from constant_velocity_moe import ConstantVelocityPrototypeMoE

def main():
    payload=torch.load(ROOT/'outputs/constant_velocity_moe_development/constant_velocity_moe.pth',map_location='cuda' if torch.cuda.is_available() else 'cpu',weights_only=False)
    spec=payload['protocol']; ms=spec['model']; device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    paths=sorted((ROOT/'data/observations/eikonal_independent').glob('*.npz'))+sorted((ROOT/'data/observations/eikonal_wide_search_independent').glob('*.npz'))
    first=np.load(paths[0]); receivers=first['receiver_coords'].astype(np.float32); domain=np.asarray([*ms['spatial_lengths_m'],ms['time_length_s']],np.float32)
    model=ConstantVelocityPrototypeMoE(receivers,tuple(ms['spatial_lengths_m']),spec['vavg_m_s'],time_length_s=ms['time_length_s'],frequency_hz=ms['frequency_hz'],expert_count=ms['expert_count'],hidden_dim=ms['hidden_dim'],hidden_layers=ms['hidden_layers'],prototype_centers=torch.tensor(ms['prototype_centers']),prototype_scales=tuple(ms['prototype_scales']),temperature=ms['prototype_temperature']).to(device); model.load_state_dict(payload['state_dict']); model.eval()
    rows=[]
    for path in paths:
        a=np.load(path); data=a['data'].astype(np.float32); times=a['times'].astype(np.float32); truth=a['source_truth'].astype(float); rid=np.arange(len(receivers))[::2]; tid=np.arange(len(times))[::8]; rr=receivers[rid]; tt=times[tid]; pts=np.concatenate((np.broadcast_to(rr[:,None,:],(len(rid),len(tid),3)),np.broadcast_to(tt[None,:,None],(len(rid),len(tid),1))),-1).reshape(-1,4)/domain; x=torch.as_tensor(pts,device=device,dtype=torch.float32); idx=torch.as_tensor(np.repeat(rid,len(tid)),device=device); obs=torch.as_tensor(data[np.ix_(rid,tid)],device=device); rms=float(np.sqrt(np.mean(data.astype(float)**2))); vals=[]
        with torch.inference_mode():
            for v in [3000.,3400.,3800.,4200.,4600.,5000.,5400.]:
                model.vavg=float(v); src=torch.as_tensor(np.broadcast_to(truth/domain,(len(pts),4)).copy(),device=device,dtype=torch.float32); pred=model(x,src,idx)[0].reshape(len(rid),len(tid)); vals.append({'vavg_m_s':v,'truth_source_loss':float(((pred-obs)/rms).square().mean())})
        rows.append({'event':path.name,'values':vals})
    summary=[]
    for v in [3000.,3400.,3800.,4200.,4600.,5000.,5400.]: summary.append({'vavg_m_s':v,'mean_truth_source_loss':float(np.mean([next(z['truth_source_loss'] for z in r['values'] if z['vavg_m_s']==v) for r in rows]))})
    out={'model_checkpoint':'outputs/constant_velocity_moe_development/constant_velocity_moe.pth','diagnostic':'frozen-model vavg sensitivity; no retraining','events':rows,'summary':summary}; p=ROOT/'experiments/constant_velocity_vavg_development_sensitivity.json'; p.write_text(json.dumps(out,indent=2)); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
