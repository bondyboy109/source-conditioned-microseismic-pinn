"""Evaluate the constant-velocity MoE on the locked spatial-refinement events."""
from __future__ import annotations
import itertools, json, sys, time
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from constant_velocity_moe import ConstantVelocityPrototypeMoE
from examples.invert_fmm_receiver_fields import receiver_fields, spatial_grid, interpolate_fields
from examples.audit_arrival_spatial_resolution import rank_candidates

PLAN=ROOT/'experiments/refined_fmm_moe_diagnostic_protocol.json'
def pool(fields,picks,spec):
    bounds=np.asarray(spec['source_bounds_m'],float); coarse=spatial_grid(bounds,tuple(spec['coarse_grid']))
    scores,_=rank_candidates(fields,picks,coarse,spec); current=np.column_stack((coarse[np.argsort(scores,kind='stable')[:3]],np.zeros(3)))
    step=(bounds[:,1]-bounds[:,0])/(np.asarray(spec['coarse_grid'])-1); offsets=np.asarray(list(itertools.product((-1,0,1),repeat=3)))
    seen=[np.column_stack((coarse,np.zeros(len(coarse))))]
    for _ in range(3):
        step/=2; trial=np.clip(current[:,None,:3]+offsets[None,:,:]*step[:3],bounds[:3,0],bounds[:3,1]).reshape(-1,3)
        cand=np.unique(np.concatenate((current[:,:3],trial)),axis=0); seen.append(np.column_stack((cand,np.zeros(len(cand)))))
        rm,_=rank_candidates(fields,picks,cand,spec); current=np.column_stack((cand[np.argsort(rm,kind='stable')[:3]],np.zeros(3)))
    out=np.unique(np.concatenate(seen),axis=0)
    return out
def main():
    plan=json.loads(PLAN.read_text()); reg=json.loads((ROOT/plan['source_protocol']).read_text()); manifest=json.loads((ROOT/reg['manifest']).read_text())
    spec=json.loads((ROOT/'experiments/arrival_constrained_fresh_test_protocol.json').read_text())
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); payload=torch.load(ROOT/'outputs/constant_velocity_moe_development/constant_velocity_moe.pth',map_location=device,weights_only=False); ms=payload['protocol']['model']; first_path=ROOT/reg['test_directory']/(reg['test_events'][0]['id']+'.npz'); receivers=np.load(first_path)['receiver_coords'].astype(np.float32)
    fields=receiver_fields(np.asarray(np.load(ROOT/'data/velocity/vp_model.npy',mmap_mode='r'))[::2,::2,::2],receivers,80.0)
    model=ConstantVelocityPrototypeMoE(receivers,tuple(ms['spatial_lengths_m']),payload['protocol']['vavg_m_s'],time_length_s=ms['time_length_s'],frequency_hz=ms['frequency_hz'],expert_count=ms['expert_count'],hidden_dim=ms['hidden_dim'],hidden_layers=ms['hidden_layers'],prototype_centers=torch.tensor(ms['prototype_centers']),prototype_scales=tuple(ms['prototype_scales']),temperature=ms['prototype_temperature']).to(device); model.load_state_dict(payload['state_dict']); model.eval()
    domain=np.asarray([*ms['spatial_lengths_m'],ms['time_length_s']],np.float32); rows=[]
    for event in reg['test_events']:
        path=ROOT/reg['test_directory']/(event['id']+'.npz'); a=np.load(path); data=a['data'].astype(np.float32); times=a['times'].astype(np.float32); rec=a['receiver_coords'].astype(np.float32); truth=a['source_truth'].astype(float)
        candidates=pool(fields,np.zeros(len(rec)),spec)
        rid=np.arange(len(rec))[::2]; tid=np.arange(len(times))[::8]; rr=rec[rid]; tt=times[tid]; pts=np.concatenate((np.broadcast_to(rr[:,None,:],(len(rid),len(tid),3)),np.broadcast_to(tt[None,:,None],(len(rid),len(tid),1))),-1).reshape(-1,4)/domain; x=torch.as_tensor(pts,device=device); idx=torch.as_tensor(np.repeat(rid,len(tid)),device=device); obs=torch.as_tensor(data[np.ix_(rid,tid)],device=device); rms=float(np.sqrt(np.mean(data.astype(float)**2)))
        losses=[]; start=time.perf_counter()
        with torch.inference_mode():
            for c in candidates:
                s=torch.as_tensor(np.broadcast_to(c.astype(np.float32)/domain,(len(pts),4)).copy(),device=device,dtype=torch.float32); pred=model(x,s,idx)[0].reshape(len(rid),len(tid)); losses.append(float(((pred-obs)/rms).square().mean()))
        best=candidates[int(np.argmin(losses))]; err=float(np.linalg.norm(best[:3]-truth[:3])); terr=abs(float(best[3]-truth[3])); rows.append({'id':event['id'],'estimate_m_s':best.tolist(),'truth_m_s':truth.tolist(),'location_error_m':err,'origin_time_error_s':terr,'score':min(losses),'seconds':time.perf_counter()-start}); print(event['id'],err,terr,flush=True)
    result={'method':'constant_velocity_moe','vavg_m_s':payload['protocol']['vavg_m_s'],'events':rows,'summary':{'mean_location_error_m':float(np.mean([r['location_error_m'] for r in rows])),'median_location_error_m':float(np.median([r['location_error_m'] for r in rows])),'mean_origin_time_error_s':float(np.mean([r['origin_time_error_s'] for r in rows]))}}
    out=ROOT/'experiments/constant_velocity_moe_localization_results.json'; out.write_text(json.dumps(result,indent=2)); print(json.dumps(result['summary'],indent=2))
if __name__=='__main__': main()
