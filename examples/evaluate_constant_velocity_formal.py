"""Formal constant-velocity search on frozen stress records.

Uses the same spatial coarse grid as the stress protocol. Origin time is the
clipped median of picked arrivals minus constant-velocity source-receiver
travel times; the constant-velocity model ranks candidates by waveform loss.
"""
from __future__ import annotations
import json, itertools, sys
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from constant_velocity_moe import ConstantVelocityPrototypeMoE
from examples.stress_test_traveltime_aligned_moe import add_exact_awgn, retained_receivers
from examples.run_traveltime_aligned_moe_localization import pick_arrivals

def score(model,data,times,receivers,picks,domain,candidates,vavg,device):
    rid=np.arange(len(receivers))[::2]; tid=np.arange(len(times))[::8]; rr=receivers[rid]; tt=times[tid]
    pts=np.concatenate((np.broadcast_to(rr[:,None,:],(len(rid),len(tid),3)),np.broadcast_to(tt[None,:,None],(len(rid),len(tid),1))),-1).reshape(-1,4)/domain
    x=torch.as_tensor(pts,device=device,dtype=torch.float32); idx=torch.as_tensor(np.repeat(rid,len(tid)),device=device); obs=torch.as_tensor(data[np.ix_(rid,tid)],device=device); rms=float(np.sqrt(np.mean(data.astype(float)**2))); vals=[]
    with torch.inference_mode():
        for cxyz in candidates:
            dist=np.linalg.norm(receivers-cxyz[None,:],axis=1); ts=float(np.clip(np.median(picks[np.isfinite(picks)]-dist[np.isfinite(picks)]/vavg),.05,.5)); c=np.asarray([*cxyz,ts],np.float32); s=torch.as_tensor(np.broadcast_to(c/domain,(len(pts),4)).copy(),device=device); pred=model(x,s,idx)[0].reshape(len(rid),len(tid)); vals.append(float(((pred-obs)/rms).square().mean()))
    return int(np.argmin(vals)),np.asarray(vals),ts

def refined_constant_search(model,data,times,receivers,picks,domain,vavg,device):
    bounds=np.asarray([[2000.,14000.],[2000.,14000.],[100.,1500.]])
    axes=[np.linspace(a,b,n) for (a,b),n in zip(bounds,[11,11,8])]
    current=np.asarray(list(itertools.product(*axes)),np.float32)
    best_idx,losses,_=score(model,data,times,receivers,picks,domain,current,vavg,device)
    current=current[np.argsort(losses)[:3]]
    step=(bounds[:,1]-bounds[:,0])/np.asarray([10,10,7],float)
    offsets=np.asarray(list(itertools.product((-1,0,1),repeat=3)),np.float32)
    for _ in range(3):
        step/=2
        trial=np.clip(current[:,None,:]+offsets[None,:,:]*step[None,None,:],bounds[:,0],bounds[:,1]).reshape(-1,3)
        pool=np.unique(np.concatenate((current,trial)),axis=0)
        idx,vals,_=score(model,data,times,receivers,picks,domain,pool,vavg,device)
        current=pool[np.argsort(vals)[:3]]
    idx,vals,ts=score(model,data,times,receivers,picks,domain,current,vavg,device)
    return np.r_[current[idx],ts],float(vals[idx])

def main():
    proto=json.loads((ROOT/'experiments/traveltime_aligned_moe_stress_protocol.json').read_text()); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); payload=torch.load(ROOT/'outputs/constant_velocity_moe_v5000/constant_velocity_moe.pth',map_location=device,weights_only=False); ms=payload['protocol']['model']; base=ROOT/proto['confirmation_directory']; first=np.load(base/proto['events'][0]['file']); allrec=first['receiver_coords'].astype(np.float32); domain=np.array([16000.,16000.,3720.,2.],np.float32); model=ConstantVelocityPrototypeMoE(allrec,tuple(ms['spatial_lengths_m']),5000.,time_length_s=ms['time_length_s'],frequency_hz=ms['frequency_hz'],expert_count=ms['expert_count'],hidden_dim=ms['hidden_dim'],hidden_layers=ms['hidden_layers'],prototype_centers=torch.tensor(ms['prototype_centers']),prototype_scales=tuple(ms['prototype_scales']),temperature=ms['prototype_temperature']).to(device); model.load_state_dict(payload['state_dict']); model.eval()
    bounds=np.asarray([[2000.,14000.],[2000.,14000.],[100.,1500.]]); axes=[np.linspace(a,b,n) for (a,b),n in zip(bounds,[11,11,8])]; candidates=np.asarray(list(itertools.product(*axes)),np.float32); rows=[]
    for event in proto['events']:
        clean=np.load(base/event['file']); truth=clean['source_truth'].astype(float)
        for cond in proto['conditions']:
            if cond['kind']=='noise': data,_=add_exact_awgn(clean['data'],cond['snr_db'],cond['seed']+1009*proto['events'].index(event)); rec=allrec
            else: keep=retained_receivers(len(allrec),cond['level']/100.,cond['seed']); data=clean['data'][keep].astype(np.float32); rec=allrec[keep]
            event_data={'data':data,'times':clean['times'],'receivers':rec}; _,picks=pick_arrivals(event_data,{'minimum_receiver_peak_fraction_of_record':.01,'last_record_s_excluded_for_pick':.15,'pick_bias_s':-.26279182618600316});
            if np.isfinite(picks).sum()<6: rows.append({'condition_id':cond['id'],'event':event['file'],'status':'insufficient_picks'}); continue
            est,loss=refined_constant_search(model,data,clean['times'],rec,picks,domain,5000.,device); rows.append({'condition_id':cond['id'],'event':event['file'],'status':'ok','estimated_source_m_s':est.tolist(),'truth_m_s':truth.tolist(),'location_error_m':float(np.linalg.norm(est[:3]-truth[:3])),'origin_time_error_s':float(abs(est[3]-truth[3])),'waveform_loss':loss}); print(cond['id'],event['file'],rows[-1]['location_error_m'],flush=True)
    out={'method':'constant_velocity_moe_formal_coarse_search','vavg_m_s':5000,'candidate_count':len(candidates),'rows':rows}; (ROOT/'experiments/constant_velocity_formal_stress_results.json').write_text(json.dumps(out,indent=2)); print('saved',len(rows))
if __name__=='__main__': main()
