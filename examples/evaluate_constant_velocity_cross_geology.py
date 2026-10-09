"""Zero-shot constant-velocity control on the independent geology records."""
from __future__ import annotations
import itertools, json, sys
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from constant_velocity_moe import ConstantVelocityPrototypeMoE
from examples.run_traveltime_aligned_moe_localization import pick_arrivals

def score(model,data,times,receivers,picks,domain,cands,vavg,device):
    rid=np.arange(len(receivers))[::2]; tid=np.arange(len(times))[::8]; rr=receivers[rid]; tt=times[tid]; pts=np.concatenate((np.broadcast_to(rr[:,None,:],(len(rid),len(tid),3)),np.broadcast_to(tt[None,:,None],(len(rid),len(tid),1))),-1).reshape(-1,4)/domain; x=torch.as_tensor(pts,device=device,dtype=torch.float32); idx=torch.as_tensor(np.repeat(rid,len(tid)),device=device); obs=torch.as_tensor(data[np.ix_(rid,tid)],device=device); rms=float(np.sqrt(np.mean(data.astype(float)**2))); vals=[]; estimates=[]
    with torch.inference_mode():
        for cxyz in cands:
            d=np.linalg.norm(receivers-cxyz[None,:],axis=1); ts=float(np.clip(np.median(picks[np.isfinite(picks)]-d[np.isfinite(picks)]/vavg),.05,.5)); c=np.asarray([*cxyz,ts],np.float32); s=torch.as_tensor(np.broadcast_to(c/domain,(len(pts),4)).copy(),device=device); pred=model(x,s,idx)[0].reshape(len(rid),len(tid)); vals.append(float(((pred-obs)/rms).square().mean())); estimates.append(c)
    i=int(np.argmin(vals)); return estimates[i],vals[i]

def main():
    proto=json.loads((ROOT/'experiments/cross_model_geology_protocol.json').read_text()); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); payload=torch.load(ROOT/'outputs/constant_velocity_moe_v5000/constant_velocity_moe.pth',map_location=device,weights_only=False); ms=payload['protocol']['model']; first=np.load(ROOT/proto['observation_directory']/proto['events'][0]['file']); receivers=first['receiver_coords'].astype(np.float32); domain=np.array([16000.,16000.,3720.,2.],np.float32); model=ConstantVelocityPrototypeMoE(receivers,tuple(ms['spatial_lengths_m']),5000.,time_length_s=ms['time_length_s'],frequency_hz=ms['frequency_hz'],expert_count=ms['expert_count'],hidden_dim=ms['hidden_dim'],hidden_layers=ms['hidden_layers'],prototype_centers=torch.tensor(ms['prototype_centers']),prototype_scales=tuple(ms['prototype_scales']),temperature=ms['prototype_temperature']).to(device); model.load_state_dict(payload['state_dict']); model.eval(); bounds=np.asarray(proto['source_bounds_m_s'][:3]); axes=[np.linspace(a,b,n) for (a,b),n in zip(bounds,[11,11,8])]; cands=np.asarray(list(itertools.product(*axes)),np.float32); rows=[]
    for event in proto['events']:
        a=np.load(ROOT/proto['observation_directory']/event['file']); data=a['data'].astype(np.float32); times=a['times'].astype(np.float32); truth=a['source_truth'].astype(float); _,picks=pick_arrivals({'data':data,'times':times},{'minimum_receiver_peak_fraction_of_record':.01,'last_record_s_excluded_for_pick':.15,'pick_bias_s':proto['arrival_constraint']['pick_bias_s']});
        if np.isfinite(picks).sum()<6: rows.append({'event':event['name'],'status':'insufficient_picks'}); continue
        est,loss=score(model,data,times,receivers,picks,domain,cands,5000.,device); rows.append({'event':event['name'],'status':'ok','estimated_source_m_s':est.tolist(),'truth_m_s':truth.tolist(),'location_error_m':float(np.linalg.norm(est[:3]-truth[:3])),'origin_time_error_s':float(abs(est[3]-truth[3])),'waveform_loss':loss}); print(event['name'],rows[-1]['location_error_m'],flush=True)
    ok=[r for r in rows if r['status']=='ok']; result={'method':'constant_velocity_moe_zero_shot_cross_geology','vavg_m_s':5000,'candidate_count':len(cands),'events':rows,'summary':{'median_location_error_m':float(np.median([r['location_error_m'] for r in ok])),'mean_location_error_m':float(np.mean([r['location_error_m'] for r in ok])),'median_origin_time_error_s':float(np.median([r['origin_time_error_s'] for r in ok]))}}; (ROOT/'experiments/constant_velocity_cross_geology_results.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result['summary'],indent=2))
if __name__=='__main__': main()
