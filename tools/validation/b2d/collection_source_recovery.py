"""Read-only collection identity investigation; no geometry-based promotion.

V0.0.4 collection identities and local XML hypotheses remain separate. This
tool inventories evidence and missing original job bindings. It never writes
Status, invokes CARLA, or changes an existing certification result.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

VERSION = 'collection-source-recovery-v1'
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT/'outputs/diffusiondrive_v1/route_preview_v11/20261023_route_validation_v1'
DIRECTORY = re.compile(r'(?P<scenario>.+)_(?P<town>Town\w+)_Route(?P<route_id>\d+)_Weather(?P<weather_id>\d+)_(?P<timestamp>\d{2}-\d{2}-\d{2}-\d{2}-\d{2})')
RESULT_ID = re.compile(r'RouteScenario_(?P<xml_id>\d+)_rep(?P<repetition>\d+)')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4194304), b''):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def unique(rows, key):
    index = {}
    for row in rows:
        value = row[key]
        if not value or value in index:
            raise ValueError('missing_or_duplicate_'+key)
        index[value] = row
    return index


def parse_directory(name):
    match = DIRECTORY.fullmatch(name)
    if not match:
        raise ValueError('dataset_directory_identity_not_parseable')
    return match.groupdict()


def result_identity(directory, result):
    """Preserve all records; neither record index nor row order is an XML id."""
    records = result.get('_checkpoint', {}).get('records', [])
    issues, identities = [], []
    if len(records) != 1:
        issues.append('EXPECTED_ONE_RESULT_RECORD')
    for record in records:
        match = RESULT_ID.fullmatch(str(record.get('route_id', '')))
        identities.append(match.groupdict() if match else None)
        if not match:
            issues.append('RESULT_ROUTE_ID_INVALID')
        elif match['xml_id'] != directory['route_id']:
            issues.append('DIRECTORY_RESULT_ROUTE_ID_CONFLICT')
    return identities, sorted(set(issues))


def local_xml_catalog(candidates):
    """Pin XML bytes, retain anchor/scenario/weather data as hypotheses only."""
    parsed, result = {}, {}
    for key, candidate in candidates.items():
        sources = []
        for evidence in candidate['xml_evidence']:
            path = Path(evidence['path'])
            if path not in parsed:
                parsed[path] = (sha(path), ET.parse(path).getroot())
            actual_sha, tree = parsed[path]
            if actual_sha != evidence['sha256']:
                raise ValueError('xml_evidence_sha_changed:'+str(path))
            matches = []
            for ordinal, route in enumerate(tree.findall('route')):
                if route.get('id') != evidence['xml_id'] or route.get('town') != candidate['town']:
                    continue
                positions = route.findall('waypoints/position') or route.findall('waypoint')
                xyz = [[float(p.get('x')), float(p.get('y')), float(p.get('z', 0))] for p in positions]
                if xyz != candidate['xyz']:
                    continue
                matches.append(dict(route_ordinal=ordinal,
                    scenarios=[dict(attributes=s.attrib, children=[dict(tag=c.tag,attributes=c.attrib) for c in s])
                               for s in route.findall('scenarios/scenario')],
                    weather=[p.attrib for p in route.findall('weathers/weather')]))
            if not matches:
                raise ValueError('candidate_xml_anchor_identity_changed:'+key)
            sources.append(dict(**evidence, definition_records=matches,
                release_scope='V004_EVALUATION_220' if path.name=='bench2drive_0.0.4_val.xml'
                else 'OLDER_EVALUATION_220' if path.name=='bench2drive220.xml'
                else 'DEV10_EVALUATION_SUBSET' if 'dev10' in path.name else 'GENERIC_OR_LEGACY_ROUTE'))
        result[key] = dict(town=candidate['town'], candidate_start_xyz=candidate['xyz'][0],
            candidate_end_xyz=candidate['xyz'][-1], candidate_anchors_xyz=candidate['xyz'],
            xml_sources=sources, original_collection_binding_proven=False)
    return result


def make_record(prior, recovery, directory, result, result_path, input_evidence, catalog, archive_present):
    identities, issues = result_identity(directory, result)
    keys = {r['key'] for r in prior['accepted_candidates']}
    keys.update(r['key'] for r in recovery['exact_id_leads'])
    if prior.get('selected_key'):
        keys.add(prior['selected_key'])
    candidates = []
    for key in sorted(keys):
        c = catalog[key]
        candidates.append(dict(key=key, town=c['town'],
            xml_sources=[{k:e[k] for k in ['path','sha256','xml_id','release_scope']} for e in c['xml_sources']],
            anchor_reference='xml_source_catalog.json#'+key,
            selected_by_previous_proxy=key==prior.get('selected_key'),
            numeric_id_equal=any(e['xml_id']==directory['route_id'] for e in c['xml_sources']),
            geometric_screen_metrics=next((x for x in prior['accepted_candidates'] if x['key']==key),None),
            independent_collection_binding=False))
    ambiguous = prior['association_basis']=='AMBIGUOUS_LOCAL_XML_CANDIDATES'
    return dict(version=VERSION, route=prior['route'], dataset_directory=str(result_path.parent),
        directory_identity=dict(**directory, timestamp_year=None, timestamp_timezone=None),
        result_identity=dict(path=str(result_path),sha256=sha(result_path),json_pointer='/_checkpoint/records',
            records=result.get('_checkpoint',{}).get('records',[]),parsed_ids=identities,
            progress=result.get('_checkpoint',{}).get('progress'),identity_issues=issues,
            directory_result_id_consistent=not issues),
        published_archive=dict(filename=prior['route']+'.tar.gz',exact_name_in_pinned_dataset_manifest=archive_present,
            repo_revision='d84ae18cf20a88b9220df1bc15c073d8824b7ac1',content_hash_of_original_archive_rechecked=False),
        previous_proxy=dict(source_status=prior['source_status'],association_basis=prior['association_basis'],
            evidence_grade=prior['evidence_grade'],selected_key=prior['selected_key'],
            frozen_status_rows=prior['frozen_status_rows']),
        source_status='AMBIGUOUS_SOURCE' if ambiguous else 'ORIGINAL_SOURCE_NOT_RECOVERED',
        source_grade='RESULT_DIRECTORY_ID_CONSISTENT_XML_NAMESPACE_MISSING' if not issues else 'RESULT_IDENTITY_CONFLICT',
        original_source_recovered=False, ambiguity_resolved=False,
        original_route_definition=dict(xml_path=None,xml_sha256=None,xml_id=None,start_xyz=None,end_xyz=None,
            anchors_xyz=None,dense_plan_path=None,map_identity=None,planner_version=None),
        collection_task=dict(launch_script=None,job_id=None,route_file_argument=None,routes_subset=None,
            collector_commit=None,selection_seed=None,dataset_directory_binding=None),
        candidate_xml_sources=candidates,
        uncertainties=issues+['ORIGINAL_COLLECTION_XML_OR_DENSE_PLAN_MISSING','JOB_TO_ROUTE_FILE_SHA_BINDING_MISSING',
            'ORIGINAL_COLLECTOR_ENTRYPOINT_AND_VERSION_UNKNOWN','ORIGINAL_MAP_AND_PLANNER_VERSION_UNPINNED',
            'NUMERIC_ID_IS_NOT_A_GLOBAL_XML_NAMESPACE','DIRECTORY_TIMESTAMP_HAS_NO_YEAR_OR_TIMEZONE'],
        evidence=input_evidence,frame_certified_promoted=0,training_admitted=False,
        new_frame_validation_eligible=False)


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def write_jsonl(path, rows):
    with path.open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')


def run(args):
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'collection_route_identity_manifest.jsonl').exists():
        raise FileExistsError('independent_version_output_already_exists')
    names=['route_certification_manifest.jsonl','source_recovery_inventory.json','route_validation_summary.json','xml_candidates.json']
    evidence=[dict(path=str(args.inventory/n),sha256=sha(args.inventory/n)) for n in names]
    for path in [args.output/'local_pipeline_evidence.json',args.output/'public_sources/public_source_index.json']:
        if path.is_file():
            evidence.append(dict(path=str(path),sha256=sha(path)))
    priors=unique(read_jsonl(args.inventory/names[0]),'route')
    recovery=unique(json.loads((args.inventory/names[1]).read_text())['records'],'route')
    if set(priors)!=set(recovery) or len(priors)!=1100:
        raise ValueError('1100_unique_identical_route_identities_required')
    catalog=local_xml_catalog(json.loads((args.inventory/'xml_candidates.json').read_text()))
    metadata=args.dataset_root.parent/'.hfd/repo_metadata.json'
    archives={r['rfilename'] for r in json.loads(metadata.read_text())['siblings']}
    evidence.append(dict(path=str(metadata),sha256=sha(metadata)))
    rows=[]
    for route,prior in priors.items():
        directory=parse_directory(route)
        if directory['town']!=prior['town']:
            raise ValueError('inventory_town_identity_conflict:'+route)
        path=args.dataset_root/route/'result.json'
        if sha(path)!=recovery[route]['result_sha256']:
            raise ValueError('original_result_sha_changed:'+route)
        rows.append(make_record(prior,recovery[route],directory,json.loads(path.read_text()),path,
                               evidence,catalog,route+'.tar.gz' in archives))
    groups=defaultdict(list)
    for row in rows:
        d=row['directory_identity'];groups[(d['town'],d['route_id'])].append(row)
    reuse=[]
    for (town,rid),items in groups.items():
        for item in items:
            item['route_id_reuse']=dict(town=town,route_id=rid,clip_count=len(items),
                other_dataset_routes=[r['route'] for r in items if r is not item],xml_reuse_proven=False)
        if len(items)>1:
            reuse.append(dict(town=town,route_id=rid,clips=[dict(route=r['route'],
                directory_identity=r['directory_identity'],result_identity=r['result_identity']) for r in items],
                explanation='REPEATED_TASK_ID_WITH_DIFFERENT_TIMESTAMPS; exact rerun/selection mechanism unknown',
                xml_reuse_proven=False,automatic_identity_merge=False))
    write_json(args.output/'xml_source_catalog.json',dict(version=VERSION,candidates=catalog))
    write_jsonl(args.output/'collection_route_identity_manifest.jsonl',rows)
    write_jsonl(args.output/'ambiguous_source_index.jsonl',[r for r in rows if r['source_status']=='AMBIGUOUS_SOURCE'])
    write_jsonl(args.output/'route_id_reuse_index.jsonl',reuse)
    proxies=[dict(route=p['route'],scope='EXISTING_PROXY_ONLY_NOT_NEW_ORIGINAL_SOURCE',
        source_evidence_grade=p['evidence_grade'],plan_file_sha256=p['plan_file_sha256'],
        comparable_frames=p['comparable_frames'],command_agreement_rate=p['command_agreement_rate'],
        left_right_reversals=p['left_right_reversals'],continuous_mismatch_intervals=p['continuous_mismatch_intervals'],
        new_original_source_recovered=False,new_frame_validation_run=False,frame_promotion=False)
        for p in priors.values() if p['plan_status']=='PASS']
    write_jsonl(args.output/'existing_proxy_validation_references.jsonl',proxies)
    summary=dict(version=VERSION,routes=len(rows),directory_result_consistent=sum(not r['result_identity']['identity_issues'] for r in rows),
        published_archive_name_matches=sum(r['published_archive']['exact_name_in_pinned_dataset_manifest'] for r in rows),
        scenario_counts=dict(Counter(r['directory_identity']['scenario'] for r in rows)),
        town_route_id_groups=len(groups),reused_town_route_id_groups=len(reuse),repeated_extra_clips=len(rows)-len(groups),
        prior_unverified_routes=sum(r['previous_proxy']['source_status']!='ROUTE_SOURCE_VERIFIED' for r in rows),
        original_source_recovered=0,recovered_from_prior_934=0,
        ambiguous_routes=sum(r['previous_proxy']['association_basis']=='AMBIGUOUS_LOCAL_XML_CANDIDATES' for r in rows),ambiguity_resolved=0,
        ambiguous_remaining=sum(r['source_status']=='AMBIGUOUS_SOURCE' for r in rows),
        numeric_xml_id_leads=sum(any(c['numeric_id_equal'] for c in r['candidate_xml_sources']) for r in rows),
        new_frame_validation_eligible=0,new_planner_runs=0,new_frame_certified=0,training_admitted=0,
        existing_proxy_validation_references=len(proxies),training_started=False,status_overlay_written=False)
    write_json(args.output/'collection_source_recovery_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory',type=Path,default=DEFAULT_INPUT)
    parser.add_argument('--dataset-root',type=Path,default=Path('/run/media/aethelind/datasets/Bench2Drive-V0.0.4/v1'))
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())


if __name__=='__main__':
    main()
