import yaml
import re
from pathlib import Path
from pytimeloop.looptree.run import run_looptree
from copy import deepcopy
from pprint import pp

GB_USAGE_CAP = 64000 * 4000 // 16

# GB_USAGE_CAP = 64000 * 16 // 16

def change_arch_pe_dims(path: Path, new_x: int, new_y: int):
    text = path.read_text()
    text = re.sub(
        r'^(?P<pre>\s*meshX:\s*)\d+',
        lambda m: f"{m.group('pre')}{new_x}",
        text,
        flags=re.MULTILINE
    )
    text = re.sub(
        r'^(?P<pre>\s*meshY:\s*)\d+',
        lambda m: f"{m.group('pre')}{new_y}",
        text,
        flags=re.MULTILINE
    )
    path.write_text(text)


def run_experiment(pe_mesh: tuple[int,int],
                   temporal_dims: dict[str,int],
                   m_values: range,
                   config_dr: str, # should contain template architecture, template mapping, workload
                   arch_yaml: str,
                   mapping_yaml: str,
                   tmp_dir: str,
                   bindings: dict[int,str],
                   print_res=True,
                   print_energy_bd=False,
                   print_mem_lat=False):
    
    tmp_dir = Path(tmp_dir)
    run_dir = tmp_dir / f"x{pe_mesh[0]:03d}y{pe_mesh[1]:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # copy original architecture
    src_arch = Path(config_dr) / arch_yaml
    arch_path = run_dir / 'architecture.yaml'
    arch_path.write_text(src_arch.read_text())

    change_arch_pe_dims(arch_path, pe_mesh[0], pe_mesh[1])

    # load and edit mapping 
    mapping_tpl = yaml.safe_load((Path(config_dr) / mapping_yaml).read_text())

    results = {}

    # run for diff tile sizes of m (number of rows of attention matrix)
    for m in m_values:
        mapping = deepcopy(mapping_tpl)
        mapper_run_dir = run_dir / f"m{m:03d}"
        mapper_run_dir.mkdir(exist_ok=True, parents=True)

        # update fused dims
        for node in mapping['mapping']['nodes']:
            if node.get('type') == 'temporal' and node.get('rank') == 'M3':
                node['tile_shape'] = m
            if node.get('type') == 'temporal' and node.get('rank') == 'P3':
                node['tile_shape'] = temporal_dims[node.get('rank')]
            if node.get('type') == 'temporal' and node.get('rank') in ('B3','H3'):
                node['tile_shape'] = 1

        # update sequential/pipeline branches
        seq_or_pl = next(n for n in mapping['mapping']['nodes']
                         if n['type'] in ('sequential','pipeline'))
        for branch in seq_or_pl['branches']:
            for loop in branch:
                if loop['type'] == 'temporal':
                    r = loop['rank']
                    if r in temporal_dims:
                        loop['tile_shape'] = temporal_dims[r]

        map_path = mapper_run_dir / f'mapping_m3_{m:03d}.yaml'
        map_path.write_text(yaml.safe_dump(mapping, sort_keys=False))

        # run Looptree
        stats = run_looptree(
            tmp_dir,
            [str(arch_path), str(Path(config_dr)/'workload.yaml'), str(map_path)],
            mapper_run_dir,
            bindings,
            call_accelergy=True
        )


        gb_usage = float(str(stats.capacity_usage['GlobalBuffer'])) / GB_USAGE_CAP
        latency = getattr(stats, 'latency', None) or getattr(stats, 'total_cycles', None)
        total_energy = sum((getattr(stats, 'energy', {}) or {}).values())
        
        results[m] = {
            'GB_usage':     gb_usage,
            'latency':      latency,
            'energy_total': total_energy,
            'energy_breakdown': (getattr(stats, 'energy', {})),
            'memory_latency': (getattr(stats, 'memory_latency', {}))
        }

        if print_res:
            print(f"M3={m:3d}  GB={gb_usage:.4f}  lat={latency}  E={total_energy:.3e}")
        if print_energy_bd:
            pp((getattr(stats, 'energy', {})))
        if print_mem_lat:
            pp((getattr(stats, 'memory_latency', {})))

    return results
