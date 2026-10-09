"""Evaluate the frozen constant-velocity control on existing stress records."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from constant_velocity_moe import ConstantVelocityPrototypeMoE

def main():
    stress=json.loads((ROOT/'experiments/traveltime_aligned_moe_stress_results.json').read_text())
    proto=json.loads((ROOT/'experiments/traveltime_aligned_moe_stress_protocol.json').read_text())
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); payload=torch.load(ROOT/'outputs/constant_velocity_moe_v5000/constant_velocity_moe.pth',map_location=device,weights_only=False); ms=payload['protocol']['model']; base=ROOT/proto['confirmation_directory']; first=np.load(base/proto['events'][0]['file']); receivers=first['receiver_coords'].astype(np.float32); domain=np.array([16000.,16000.,3720.,2.],np.float32)
    model=ConstantVelocityPrototypeMoE(receivers,tuple(ms['spatial_lengths_m']),5000.,time_length_s=ms['time_length_s'],frequency_hz=ms['frequency_hz'],expert_count=ms['expert_count'],hidden_dim=ms['hidden_dim'],hidden_layers=ms['hidden_layers'],prototype_centers=torch.tensor(ms['prototype_centers']),prototype_scales=tuple(ms['prototype_scales']),temperature=ms['prototype_temperature']).to(device); model.load_state_dict(payload['state_dict']); model.eval()
    rows=[]
    for event in proto['events']:
        clean=np.load(base/event['file']); truth=clean['source_truth'].astype(float)
        for cond in proto['conditions']:
            if cond['kind']=='noise':
                rng=np.random.default_rng(cond['seed']+int(event['file'].split('_')[1])); data=clean['data'].astype(np.float32); noise=rng.normal(size=data.shape).astype(np.float32); noise*=np.sqrt(np.mean(data.astype(float)**2)/(10**(cond['snr_db']/10)*np.mean(noise.astype(float)**2))); data=data+noise; rec=clean['receiver_coords'].astype(np.float32)
            else:
                rng=np.random.default_rng(cond['seed']); keep=np.sort(rng.choice(len(receivers),size=int(round(len(receivers)*(1-cond['level']/100))),replace=False)); data=clean['data'][keep].astype(np.float32); rec=receivers[keep]
            times=clean['times'].astype(np.float32); rid=np.arange(len(rec))[::2]; tid=np.arange(len(times))[::8]; rr=rec[rid]; tt=times[tid]; pts=np.concatenate((np.broadcast_to(rr[:,None,:],(len(rid),len(tid),3)),np.broadcast_to(tt[None,:,None],(len(rid),len(tid),1))),-1).reshape(-1,4)/domain; x=torch.as_tensor(pts,device=device,dtype=torch.float32); idx=torch.as_tensor(np.repeat(rid,len(tid)),device=device); obs=torch.as_tensor(data[np.ix_(rid,tid)],device=device); rms=float(np.sqrt(np.mean(data.astype(float)**2))); best=(float('inf'),None)
            with torch.inference_mode():
                for xyz in ([4000,4000,300],[8000,8000,700],[12000,12000,1100]):
                    for ts in [.1,.25,.4]:
                        c=np.asarray([*xyz,ts],np.float32); s=torch.as_tensor(np.broadcast_to(c/domain,(len(pts),4)).copy(),device=device); pred=model(x,s,idx)[0].reshape(len(rid),len(tid)); loss=float(((pred-obs)/rms).square().mean());
                        if loss<best[0]: best=(loss,c)
            rows.append({'condition_id':cond['id'],'event':event['file'],'location_error_m':float(np.linalg.norm(best[1][:3]-truth[:3])),'origin_time_error_s':abs(float(best[1][3]-truth[3])),'status':'ok'})
    result={'method':'constant_velocity_moe_v5000','scope':'exploratory stress diagnostic using fixed coarse candidates','rows':rows}; (ROOT/'experiments/constant_velocity_stress_results.json').write_text(json.dumps(result,indent=2)); print('rows',len(rows))
if __name__=='__main__': main()
