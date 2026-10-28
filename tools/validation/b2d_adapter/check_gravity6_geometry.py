"""Read-only real geometry replay of the four isolated gravity6 prototype pairs."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'repositories/DiffusionDrive'))


def main():
    import numpy as np
    import torch
    from shapely.geometry import MultiPoint
    from tools.b2d_adapter import FrameReader,RouteIndex
    from tools.b2d_adapter.lidar_decode import decode_lidar_in_process
    from tools.training.diffusiondrive_v1.runtime import native_configuration,model_configuration
    from tools.training.diffusiondrive_v1.gravity6_recovery import Gravity6Dataset,build_pair,ReferenceReader
    from tools.training.diffusiondrive_v1.full_recovery_rules import rear6,world_plane6,plane6_world,verified_actor
    from tools.training.diffusiondrive_v1.features_cache import file_digest,durable_json
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    cfg=json.loads((ROOT/'tools/training/diffusiondrive_v1/targets_full.json').read_text())
    native,model=native_configuration(cfg),model_configuration(cfg);g=native.geometry
    manifest=json.loads(args.manifest.read_text());dataset=Gravity6Dataset(manifest);torch.set_num_threads(1)
    results=[]
    for i,row in enumerate(manifest['rows']):
        features,targets=dataset[i];payload=torch.load(Path(manifest['root'])/row['relative_path'],map_location='cpu',weights_only=True)
        for item in payload['sources']:
            assert file_digest(Path(cfg['dataset_root'])/item['path'])==item['sha256']
        reader=FrameReader(RouteIndex.from_dataset_root(cfg['dataset_root'],row['route']));ref=reader.index.ref(row['frame']);raw=reader.read(ref)
        # Independent annotation transform matrix rather than the production rear6 formula.
        ego_to_world=np.linalg.inv(np.asarray(raw.ego_box['world2ego'],float))
        expected_rear=(ego_to_world@np.array([g.axle_dx_m,g.axle_dy_m,0.,1.]))[:3]
        rear_error=float(np.max(np.abs(expected_rear-rear6(raw,g))));assert rear_error<1e-3
        yaw=raw.actor_pose.yaw;c,s=math.cos(yaw),math.sin(yaw)
        basis=np.array([[c,s,0.],[s,-c,0.],[0.,0.,1.]])
        raw_xyz,_=decode_lidar_in_process(ref,cfg['lidar_decoder_python'])
        world=np.c_[raw_xyz,np.ones(len(raw_xyz))]@ego_to_world.T
        local=(world[:,:3]-expected_rear)@basis
        keep=(local[:,2]>g.lidar_split_height_m)&(local[:,2]<g.lidar_max_height_m)
        edges=np.linspace(-32,32,257);hist=np.histogramdd(local[keep,:2],bins=(edges,edges))[0]
        independent_lidar=(np.minimum(hist,5)/5).astype(np.float32)[None]
        # Annotation matrices are float precision; count exact replay separately below.
        lidar_changed=int(np.count_nonzero(independent_lidar!=features['lidar_feature'].numpy()))
        lidar_transform_error=float(np.max(np.abs(local-world_plane6(world[:,:3],raw,g))))
        assert lidar_transform_error<1e-3
        inverse_error=float(np.max(np.abs(plane6_world(world_plane6(world[:4096,:3],raw,g),raw,g)-world[:4096,:3])))
        assert inverse_error<1e-8
        history=[reader.read(reader.index.ref(fid)) for fid in range(row['frame']-2,row['frame']+1)]
        rear_points=[];rotation_only_rears=[];translation_discrepancies=[]
        for f in history:
            m=np.linalg.inv(np.asarray(f.ego_box['world2ego'],float))
            rear_points.append((m@np.array([g.axle_dx_m,g.axle_dy_m,0.,1.]))[:3])
            rotation_only_rears.append(np.asarray(f.ego_box['location'])+m[:3,:3]@np.array([g.axle_dx_m,g.axle_dy_m,0.]))
            translation_discrepancies.append(float(np.max(np.abs(m[:3,3]-f.ego_box['location']))))
        rear_points=np.array(rear_points);dt=g.source_dt_s
        v=(rear_points[-1]-rear_points[-2])/dt;a=(rear_points[-1]-2*rear_points[-2]+rear_points[-3])/dt**2
        kinematics=np.r_[(v@basis)[:2],(a@basis)[:2]]
        status_error=float(np.max(np.abs(kinematics-features['status_feature'].numpy()[4:])))
        # Matrix translation and explicit location are independently rounded in
        # this source. Record their finite-difference amplification; validate
        # the specified location-based contract using independent matrix R.
        rr=np.asarray(rotation_only_rears)
        rv=(rr[-1]-rr[-2])/dt;ra=(rr[-1]-2*rr[-2]+rr[-3])/dt**2
        rotation_only_error=float(np.max(np.abs(np.r_[(rv@basis)[:2],(ra@basis)[:2]]-features['status_feature'].numpy()[4:])))
        assert max(translation_discrepancies)<1e-3 and rotation_only_error<1e-3
        near=np.array([raw.navigation['x_command_near'],raw.navigation['y_command_near'],expected_rear[2]])
        bearing=math.degrees(math.atan2(*((near-expected_rear)@basis)[[1,0]]))
        category=0 if bearing>native.navigation_bearing_threshold_deg else 2 if bearing<-native.navigation_bearing_threshold_deg else 1
        assert np.array_equal(np.eye(4)[category],features['status_feature'].numpy()[:4])
        trajectory_errors=[];yaw_errors=[]
        for j,fid in enumerate(range(row['frame']+5,row['frame']+41,5)):
            future=reader.read(reader.index.ref(fid));m=np.linalg.inv(np.asarray(future.ego_box['world2ego'],float))
            point=(m@np.array([g.axle_dx_m,g.axle_dy_m,0.,1.]))[:3]
            expected=(point-expected_rear)@basis
            trajectory_errors.append(float(np.max(np.abs(expected[:2]-targets['trajectory'][j,:2].numpy()))))
            expected_yaw=(yaw-future.actor_pose.yaw+math.pi)%(2*math.pi)-math.pi
            yaw_errors.append(abs(float(targets['trajectory'][j,2])-expected_yaw))
        assert max(trajectory_errors)<1e-3 and max(yaw_errors)<1e-5
        derived=ReferenceReader(reader,g).read(ref);hull_errors=[];actor_count=0;independent_detection=[]
        for box in raw.annotation['bounding_boxes']:
            if box.get('class')!='vehicle':continue
            heading=(yaw-math.radians(box['rotation'][2])+math.pi)%(2*math.pi)-math.pi
            center=((np.asarray(box['center'])-expected_rear)@basis)[:2]
            state=np.r_[center,heading,2*np.asarray(box['extent'])[:2]]
            if np.max(np.abs(box['rotation'][:2]))<=g.max_planar_tilt_deg:
                if model.lidar_min_x<=state[0]<=model.lidar_max_x and model.lidar_min_y<=state[1]<=model.lidar_max_y:independent_detection.append(state)
                continue
            hull,state,proof=verified_actor(box,derived,g)
            corners=np.asarray(box['world_cord'],float);expected_hull=MultiPoint(((corners-expected_rear)@basis)[:,:2]).convex_hull
            hull_errors.append(float(hull.hausdorff_distance(expected_hull)));actor_count+=1
            # Corner-derived rectangle, independently of the production box vertices.
            hcos,hsin=math.cos(heading),math.sin(heading);rotation=np.array([[hcos,-hsin],[hsin,hcos]])
            aligned=np.asarray(expected_hull.exterior.coords)[:-1]@rotation
            low,high=aligned.min(0),aligned.max(0)
            state=np.r_[((low+high)/2)@rotation.T,heading,high-low]
            if model.lidar_min_x<=state[0]<=model.lidar_max_x and model.lidar_min_y<=state[1]<=model.lidar_max_y:independent_detection.append(state)
        assert max(hull_errors,default=0)<.01
        states=np.array(independent_detection,dtype=np.float32).reshape(-1,5)
        if len(states):states=states[np.argsort(np.linalg.norm(states[:,:2],axis=1))[:model.num_bounding_boxes]]
        actual_states=targets['agent_states'][targets['agent_labels']].numpy()
        assert states.shape==actual_states.shape
        detection_error=float(np.max(np.abs(states-actual_states))) if len(states) else 0.
        assert detection_error<1e-3
        # Replay all heads from actual JPG/LAZ/annotation/map without writing any cache.
        rebuilt_features,rebuilt_targets,proof=build_pair(cfg,dict(row),native,model)
        feature_equal={k:torch.equal(v,rebuilt_features[k]) for k,v in features.items()}
        target_equal={k:torch.equal(v,rebuilt_targets[k]) for k,v in targets.items()}
        assert all(feature_equal.values()) and all(target_equal.values())
        assert file_digest(proof['source_map']['map_source'])==proof['source_map']['map_sha256']
        results.append({'sample_id':row['sample_id'],'route':row['route'],'frame':row['frame'],
            'rotation_deg':raw.ego_box['rotation'],'raw_sources_sha_verified':len(payload['sources']),
            'rear_vs_annotation_matrix_max_m':rear_error,'lidar_independent_transform_max_m':lidar_transform_error,
            'lidar_annotation_matrix_histogram_changed_cells':lidar_changed,'world_inverse_max_m':inverse_error,
            'status_kinematics_vs_matrix_max':status_error,'navigation_onehot_matches':True,
            'matrix_translation_vs_location_max_m':max(translation_discrepancies),
            'status_location_plus_independent_matrix_rotation_max':rotation_only_error,
            'trajectory_xy_vs_matrix_max_m':max(trajectory_errors),'trajectory_heading_max_rad':max(yaw_errors),
            'projected_actor_count':actor_count,'actor_hull_vs_world_cord_max_m':max(hull_errors,default=0),
            'independent_selected_detections':len(states),'detection_state_vs_independent_matrix_max':detection_error,
            'all_feature_replay_equal':feature_equal,'all_target_replay_equal':target_equal,
            'map_sha_verified':True,'bev_reference':'Source map world geometry and actors projected at the same true rear as LiDAR/trajectory.'})
    durable_json(args.output,{'producer_sha256':manifest['producer_sha256'],'count':len(results),'cases':results,
        'cache_written':False,'full_regeneration_started':False,
        'limits':['Four prototypes are from one Town11 Route, not cross-Town coverage.',
            'Frame cadence is the existing dataset clock contract; absolute timestamps are not independently certified.',
            'Independent transforms use annotation world2ego matrices; this is engineering consistency, not sensor/map physical truth certification.',
            'Small differences from rounded annotation matrices can move points across histogram bin boundaries.']})
    print(json.dumps({'passed':True,'cases':len(results),'output':str(args.output)}))


if __name__=='__main__':main()
