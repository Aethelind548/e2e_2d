"""Finite cross-map/rare-class inspection using the selected fit configuration."""
import json
import argparse
from pathlib import Path
import sys
import os

ROOT=Path(__file__).resolve().parents[3]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'repositories/DiffusionDrive'))
os.environ.setdefault('MPLCONFIGDIR','/tmp/e2e_v1_mpl')


def main():
    import torch
    from tools.training.diffusiondrive_v1.runtime import fit_configuration,model_configuration,split_datasets
    from tools.b2d_adapter.v1_bev import bev_annotations
    from tools.b2d_adapter.v1_builders import official_v1_module
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('m1_subset.json'))
    args=parser.parse_args();config,groups=fit_configuration(args.config);model=model_configuration(config)
    data,provider=split_datasets(config,model)
    output=ROOT/'analysis/diffusiondrive_b2d_architecture/v1_validation/m1';output.mkdir(parents=True,exist_ok=True)
    report={'status':'running','route_groups':{k:sorted(v) for k,v in groups.items()},'samples':[]}
    try:
        labels_seen=set()
        for split,concat in data.items():
            for dataset in concat.datasets:
                for ref in dataset.refs:
                    result=dataset.target_builder.build_targets(ref)
                    again=dataset.target_builder.build_targets(ref)
                    labels=result.tensors['bev_semantic_map']
                    assert torch.equal(labels,again.tensors['bev_semantic_map'])
                    assert again.provenance['bev']['cache_read']=='hit'
                    raw=dataset.reader.read(ref);anno,_=bev_annotations(raw,dataset.feature_builder.config.geometry)
                    builder=official_v1_module('transfuser_features').TransfuserTargetBuilder(model)
                    # Use each genuine rare-class box's official projected mask as reference.
                    checks={}
                    for label in (4,6):
                        layers=model.bev_semantic_classes[label][1]
                        reference=builder._compute_box_mask(anno,layers)
                        count=int(reference.sum());matched=int((labels[torch.from_numpy(reference)]==label).sum())
                        if count:
                            assert matched>0,'real rare target did not survive class projection'
                            labels_seen.add(label)
                        checks[str(label)]={'real_box_mask_pixels':count,'final_matching_pixels':matched}
                    counts=torch.bincount(labels.flatten(),minlength=7).tolist()
                    assert counts[1]+counts[3]>0
                    report['samples'].append({'split':split,'token':ref.token,'bev':result.provenance['bev'],
                                             'rare_class_checks':checks,'valid_vehicle_count':int(result.tensors['agent_labels'].sum())})
        assert labels_seen=={4,6},'selected examples must cover real static and pedestrian pixels'
        report.update(status='pass',sample_count=len(report['samples']),cache_stats=provider.cache_stats,
                      real_rare_classes_verified=sorted(labels_seen))
    except Exception as exc:
        report.update(status='failed',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        (output/'labels.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'samples':report['sample_count'],'cache':report['cache_stats']},ensure_ascii=False))


if __name__=='__main__':main()
