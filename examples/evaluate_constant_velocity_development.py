"""Evaluate trained constant-velocity controls on the 16-event development set."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from constant_velocity_moe import ConstantVelocityPrototypeMoE
from examples.validate_arrival_constrained import pick_receivers
from examples.audit_arrival_spatial_resolution import rank_candidates
from examples.invert_fmm_receiver_fields import receiver_fields, spatial_grid, interpolate_fields

def main():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    paths=sorted((ROOT/'data/observations/eikonal_independent').glob('*.npz'))+sorted((ROOT/'data/observations/eikonal_wide_search_independent').glob('*.npz'))
    first=np.load(paths[0]); receivers=first['receiver_coords'].astype(np.float32); domain=np.array([16000.,16000.,3720.,2.],np.float32); rows=[]
    fspec=json.loads((ROOT/'experiments/arrival_constrained_fresh_test_protocol.json').read_text()); vel=np.asarray(np.load(ROOT/fspec['velocity'],mmap_mode='r'))[::fspec['fmm_velocity_stride'],::fspec['fmm_velocity_stride'],::fspec['fmm_velocity_stride']]; fields=receiver_fields(vel,receivers,fspec['fmm_spacing_m'])
    for v in (3000,5000):
        payload=torch.load(ROOT/f'outputs/constant_velocity_moe_v{v}/constant_velocity_moe.pth',map_location=device,weights_only=False); ms=payload['protocol']['model']; model=ConstantVelocityPrototypeMoE(receivers,tuple(ms['spatial_lengths_m']),v,time_length_s=ms['time_length_s'],frequency_hz=ms['frequency_hz'],expert_count=ms['expert_count'],hidden_dim=ms['hidden_dim'],hidden_layers=ms['hidden_layers'],prototype_centers=torch.tensor(ms['prototype_centers']),prototype_scales=tuple(ms['prototype_scales']),temperature=ms['prototype_temperature']).to(device); model.load_state_dict(payload['state_dict']); model.eval()
        for path in paths:
            a=np.load(path); data=a['data'].astype(np.float32); times=a['times'].astype(np.float32); truth=a['source_truth'].astype(float); picks=pick_receivers(data,times,'robust_mad_6sigma',fspec); valid=np.isfinite(picks); coarse=spatial_grid(np.asarray(fspec['source_bounds_m']),tuple(fspec['coarse_grid'])); rm,_=rank_candidates(fields,picks,coarse,fspec); step=(np.asarray(fspec['source_bounds_m'])[:,1]-np.asarray(fspec['source_bounds_m'])[:,0])/(np.asarray(fspec['coarse_grid'])-1); current=coarse[np.argsort(rm)[:3]]; seen=[coarse]
            for _ in range(3):
                step/=2; off=np.asarray(list(__import__('itertools').product((-1,0,1),repeat=3))); trial=np.clip(current[:,None,:]+off[None,:,:]*step[:3],np.asarray(fspec['source_bounds_m'])[:3,0],np.asarray(fspec['source_bounds_m'])[:3,1]).reshape(-1,3); cand=np.unique(np.concatenate((current,trial)),axis=0); seen.append(cand); rr,_=rank_candidates(fields,picks,cand,fspec); current=cand[np.argsort(rr)[:3]]
            candidates=np.unique(np.concatenate(seen),axis=0); rid=np.arange(len(receivers))[::2]; tid=np.arange(len(times))[::8]; rr=receivers[rid]; tt=times[tid]; pts=np.concatenate((np.broadcast_to(rr[:,None,:],(len(rid),len(tid),3)),np.broadcast_to(tt[None,:,None],(len(rid),len(tid),1))),-1).reshape(-1,4)/domain; x=torch.as_tensor(pts,device=device,dtype=torch.float32); idx=torch.as_tensor(np.repeat(rid,len(tid)),device=device); obs=torch.as_tensor(data[np.ix_(rid,tid)],device=device); rms=float(np.sqrt(np.mean(data.astype(float)**2))); best=(float('inf'),None)
            with torch.inference_mode():
                for cxyz in candidates:
                    ts=float(np.clip(np.median(picks[valid]-np.linalg.norm(receivers[valid]-cxyz[None,:],axis=1)/v),.05,.5)); c=np.asarray([*cxyz,ts],np.float32); s=torch.as_tensor(np.broadcast_to(c/domain,(len(pts),4)).copy(),device=device); pred=model(x,s,idx)[0].reshape(len(rid),len(tid)); loss=float(((pred-obs)/rms).square().mean());
                    if loss<best[0]: best=(loss,c)
            err=float(np.linalg.norm(best[1][:3]-truth[:3])); terr=abs(float(best[1][3]-truth[3])); rows.append({'vavg_m_s':v,'event':path.name,'location_error_m':err,'origin_time_error_s':terr,'score':best[0]})
    summary=[]
    for v in (3000,5000):
        sub=[r for r in rows if r['vavg_m_s']==v]; summary.append({'vavg_m_s':v,'mean_location_error_m':float(np.mean([r['location_error_m'] for r in sub])),'median_location_error_m':float(np.median([r['location_error_m'] for r in sub])),'mean_origin_time_error_s':float(np.mean([r['origin_time_error_s'] for r in sub]))})
    out={'purpose':'development-only candidate comparison','events':rows,'summary':summary,'limitations':'This evaluator uses a coarse diagnostic candidate set; use the registered search for final claims.'}; (ROOT/'experiments/constant_velocity_development_results.json').write_text(json.dumps(out,indent=2)); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
