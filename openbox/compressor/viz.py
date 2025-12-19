import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from ConfigSpace import ConfigurationSpace
import json
import os

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def visualize_compression_details(compressor, save_dir: str):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    
    os.makedirs(save_dir, exist_ok=True)
    
    pipeline = compressor.pipeline
    if pipeline is None:
        print("  No pipeline found in compressor")
        return
    
    for i, step in enumerate(pipeline.steps):
        if hasattr(step, 'compression_info') and step.compression_info:
            info = step.compression_info
            
            if 'compressed_params' in info and len(info['compressed_params']) > 0:
                fig = plt.figure(figsize=(14, max(8, len(info['compressed_params']) * 0.4)))
                
                compressed_params = info['compressed_params']
                n_params = len(compressed_params)
                
                if n_params > 30:
                    compressed_params = compressed_params[:30]
                    n_params = 30
                
                param_names = [p['name'].split('.')[-1] for p in compressed_params]  # shorten label
                
                original_ranges = []
                compressed_ranges = []
                compression_ratios = []
                param_labels = []
                
                for param in compressed_params:
                    if 'original_range' in param:
                        original_ranges.append(param['original_range'])
                        compressed_ranges.append(param['compressed_range'])
                        compression_ratios.append(param['compression_ratio'])
                        
                        if 'original_num_values' in param:
                            label = f"{param['original_num_values']}→{param['quantized_num_values']} values"
                            param_labels.append(label)
                        else:
                            param_labels.append('')
                
                y_pos = np.arange(n_params)
                
                ax = plt.subplot(111)
                
                for idx, (orig, comp, name, label) in enumerate(zip(original_ranges, compressed_ranges, param_names, param_labels)):
                    orig_min, orig_max = orig[0], orig[1]
                    comp_min, comp_max = comp[0], comp[1]
                    
                    norm_orig_start = 0.0
                    norm_orig_end = 1.0
                    
                    is_quantization = label != ''  # quantiazation if has label
                    
                    if is_quantization:
                        norm_comp_start = 0.0
                        norm_comp_end = 1.0
                        edge_style = 'dashed'
                    else:
                        if orig_max - orig_min > 0:
                            norm_comp_start = (comp_min - orig_min) / (orig_max - orig_min)
                            norm_comp_end = (comp_max - orig_min) / (orig_max - orig_min)
                        else:
                            norm_comp_start = 0.0
                            norm_comp_end = 1.0
                        edge_style = 'solid'
                    
                    ax.barh(idx, norm_orig_end - norm_orig_start, left=norm_orig_start, height=0.4, 
                           alpha=0.3, color='gray', label='Original' if idx == 0 else '')
                    
                    ratio = compression_ratios[idx]
                    color = plt.cm.RdYlGn_r(ratio)  # greener if has smaller compress ratio
                    
                    if is_quantization:
                        ax.barh(idx, norm_comp_end - norm_comp_start, left=norm_comp_start, height=0.4,
                               alpha=0.5, color=color, edgecolor=color, linewidth=2, linestyle='--',
                               label='Quantized (mapped)' if idx == 0 and label else '')
                    else:
                        ax.barh(idx, norm_comp_end - norm_comp_start, left=norm_comp_start, height=0.4,
                               alpha=0.8, color=color, label='Compressed' if idx == 0 and not label else '')
                    
                    if label:  # 量化参数
                        ax.text(1.02, idx, f'{ratio:.1%} ({label})', 
                               va='center', fontsize=7)
                        ax.text(0.5, idx - 0.35, f'→[{int(comp_min)}, {int(comp_max)}]', 
                               va='top', ha='center', fontsize=6, color='black', 
                               fontweight='bold', style='italic')
                    else:  # 范围压缩参数
                        ax.text(1.02, idx, f'{ratio:.1%}', 
                               va='center', fontsize=8)
                        ax.text(0.5, idx - 0.35, f'→[{comp_min:.0f}, {comp_max:.0f}]', 
                               va='top', ha='center', fontsize=6, color='black', 
                               fontweight='bold')
                    
                    ax.text(-0.05, idx + 0.25, f'{orig_min:.0f}', 
                           va='center', ha='right', fontsize=7, color='gray', alpha=0.6)
                    ax.text(1.05, idx + 0.25, f'{orig_max:.0f}', 
                           va='center', ha='left', fontsize=7, color='gray', alpha=0.6)
                
                ax.set_yticks(y_pos)
                ax.set_yticklabels(param_names, fontsize=9)
                ax.set_xlim(-0.15, 1.25)
                ax.set_xlabel('Normalized Range [0=lower, 1=upper]', fontsize=12, fontweight='bold')
                ax.set_title(f'{step.name}: Range Compression Details (Top {n_params} params)', 
                           fontsize=14, fontweight='bold', pad=20)
                ax.legend(loc='upper right')
                ax.grid(axis='x', alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'range_compression_step_{i+1}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  Saved range_compression_step_{i+1}.png")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    step_names = ['Original'] + [step.name for step in pipeline.steps]
    dimensions = [len(space.get_hyperparameters()) for space in pipeline.space_after_steps]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(dimensions)))
    bars = ax.bar(range(len(dimensions)), dimensions, color=colors, alpha=0.8, edgecolor='black')
    
    for bar, dim in zip(bars, dimensions):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(dim)}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xticks(range(len(step_names)))
    ax.set_xticklabels(step_names, rotation=45, ha='right')
    ax.set_ylabel('Number of Parameters', fontsize=11, fontweight='bold')
    ax.set_title('Dimension Reduction Across Steps', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    ax = axes[0, 1]
    compression_ratios = [dim / dimensions[0] for dim in dimensions[1:]]
    step_names_no_orig = step_names[1:]
    
    colors = plt.cm.RdYlGn_r(compression_ratios)
    bars = ax.bar(range(len(compression_ratios)), compression_ratios, color=colors, alpha=0.8, edgecolor='black')
    
    for bar, ratio in zip(bars, compression_ratios):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{ratio:.1%}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xticks(range(len(step_names_no_orig)))
    ax.set_xticklabels(step_names_no_orig, rotation=45, ha='right')
    ax.set_ylabel('Compression Ratio', fontsize=11, fontweight='bold')
    ax.set_title('Compression Ratio by Step', fontsize=12, fontweight='bold')
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='No compression')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    ax = axes[1, 0]
    range_stats = []
    step_labels = []
    
    for i, step in enumerate(pipeline.steps):
        if hasattr(step, 'compression_info') and step.compression_info:
            info = step.compression_info
            if 'compressed_params' in info:
                n_compressed = len(info['compressed_params'])
                n_unchanged = len(info.get('unchanged_params', []))
                range_stats.append([n_compressed, n_unchanged])
                step_labels.append(f"Step {i+1}\n{step.name}")
    
    if range_stats:
        range_stats = np.array(range_stats)
        x = np.arange(len(step_labels))
        width = 0.35
        
        ax.bar(x, range_stats[:, 0], width, label='Compressed', alpha=0.8, color='coral')
        ax.bar(x, range_stats[:, 1], width, bottom=range_stats[:, 0], 
              label='Unchanged', alpha=0.8, color='lightblue')
        
        ax.set_ylabel('Number of Parameters', fontsize=11, fontweight='bold')
        ax.set_title('Range Compression Statistics', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(step_labels, fontsize=9)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No range compression', ha='center', va='center',
               transform=ax.transAxes, fontsize=14)
        ax.axis('off')
    
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = "Compression Summary\n" + "="*40 + "\n\n"
    summary_text += f"Original dimensions: {dimensions[0]}\n"
    summary_text += f"Final sample space: {len(pipeline.sample_space.get_hyperparameters())}\n"
    summary_text += f"Final surrogate space: {len(pipeline.surrogate_space.get_hyperparameters())}\n"
    summary_text += f"Overall compression: {len(pipeline.surrogate_space.get_hyperparameters())/dimensions[0]:.1%}\n\n"
    
    summary_text += "Steps:\n"
    for i, step in enumerate(pipeline.steps):
        input_dim = dimensions[i]
        output_dim = dimensions[i+1]
        dimension_ratio = output_dim / input_dim if input_dim > 0 else 1.0
        
        summary_text += f"{i+1}. {step.name}\n"
        
        if hasattr(step, 'compression_info') and step.compression_info:
            info = step.compression_info
            if 'avg_compression_ratio' in info:
                effective_ratio = info['avg_compression_ratio']
                summary_text += f"   Dim: {input_dim} → {output_dim} ({dimension_ratio:.1%})\n"
                summary_text += f"   Effective: {effective_ratio:.1%}\n"
            else:
                summary_text += f"   {input_dim} → {output_dim} ({dimension_ratio:.1%})\n"
        else:
            summary_text += f"   {input_dim} → {output_dim} ({dimension_ratio:.1%})\n"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
           fontsize=10, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'compression_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved compression_summary.png")


def visualize_dimension_reduction(steps: List[str], dimensions: List[int], save_path: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(steps))
    bars = ax.bar(x_pos, dimensions, alpha=0.7, color='steelblue')
    
    for i, (bar, dim) in enumerate(zip(bars, dimensions)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{dim}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Compression Step', fontsize=12)
    ax.set_ylabel('Number of Parameters', fontsize=12)
    ax.set_title('Dimension Reduction Through Compression Pipeline', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(steps, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved dimension reduction plot to {save_path}")


def visualize_parameter_importance(param_names: List[str], importances: List[float], save_path: str, topk: int = 20):
    sorted_indices = np.argsort(importances)[-topk:]
    top_names = [param_names[i] for i in sorted_indices]
    top_importances = [importances[i] for i in sorted_indices]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(len(top_names))
    bars = ax.barh(y_pos, top_importances, alpha=0.7, color='coral')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(f'Top-{topk} Parameter Importance', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    for i, (bar, imp) in enumerate(zip(bars, top_importances)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f'{imp:.4f}',
                ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved parameter importance plot to {save_path}")


def visualize_range_compression(original_ranges: Dict[str, Tuple[float, float]], 
                                compressed_ranges: Dict[str, Tuple[float, float]],
                                save_path: str, topk: int = 10):
    compression_ratios = {}
    for param_name in compressed_ranges:
        if param_name in original_ranges:
            orig_min, orig_max = original_ranges[param_name]
            comp_min, comp_max = compressed_ranges[param_name]
            orig_range = orig_max - orig_min
            comp_range = comp_max - comp_min
            if orig_range > 0:
                compression_ratios[param_name] = comp_range / orig_range
    
    sorted_params = sorted(compression_ratios.items(), key=lambda x: x[1])[:topk]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = np.arange(len(sorted_params))
    param_names = [p[0] for p in sorted_params]
    
    orig_widths = [original_ranges[p][1] - original_ranges[p][0] for p in param_names]
    comp_widths = [compressed_ranges[p][1] - compressed_ranges[p][0] for p in param_names]
    orig_starts = [original_ranges[p][0] for p in param_names]
    comp_starts = [compressed_ranges[p][0] for p in param_names]
    
    ax.barh(y_pos, orig_widths, left=orig_starts, alpha=0.5, label='Original Range', color='lightblue')
    ax.barh(y_pos, comp_widths, left=comp_starts, alpha=0.8, label='Compressed Range', color='coral')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(param_names)
    ax.set_xlabel('Parameter Value', fontsize=12)
    ax.set_title(f'Range Compression (Top-{topk} Most Compressed)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved range compression plot to {save_path}")


def visualize_pipeline_flow(steps: List[Dict], save_path: str):
    fig, ax = plt.subplots(figsize=(14, 6))
    
    n_steps = len(steps)
    x_positions = np.linspace(0, 10, n_steps + 1)
    
    for i in range(n_steps):
        step = steps[i]
        x_start = x_positions[i]
        x_end = x_positions[i + 1]
        y_pos = 0.5
        
        rect = plt.Rectangle((x_start - 0.3, y_pos - 0.15), 0.6, 0.3,
                            facecolor='steelblue', alpha=0.7, edgecolor='black')
        ax.add_patch(rect)
        
        ax.text(x_start, y_pos, step['name'], ha='center', va='center',
               fontsize=9, fontweight='bold', color='white')
        
        ax.text(x_start, y_pos - 0.3, f"{step['input_dim']}→{step['output_dim']}",
               ha='center', va='top', fontsize=8)
        
        if i < n_steps - 1:
            arrow = plt.Arrow(x_start + 0.3, y_pos, x_end - x_start - 0.6, 0,
                            width=0.05, facecolor='black', edgecolor='black')
            ax.add_patch(arrow)
    
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Compression Pipeline Flow', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved pipeline flow plot to {save_path}")


def save_compression_info(compressor, save_path: str):
    info = {
        'compression_info': compressor.compression_info,
        'sample_space_params': len(compressor.sample_space.get_hyperparameters()),
        'surrogate_space_params': len(compressor.surrogate_space.get_hyperparameters()),
    }
    
    if hasattr(compressor, 'pipeline') and compressor.pipeline:
        steps_info = []
        for i, step in enumerate(compressor.pipeline.steps):
            step_info = {
                'step_index': i + 1,
                'step_name': step.name,
                'step_type': step.__class__.__name__,
                'input_dim': len(step.input_space.get_hyperparameters()) if step.input_space else 0,
                'output_dim': len(step.output_space.get_hyperparameters()) if step.output_space else 0
            }
            
            if hasattr(step, '_shap_cache') and step._shap_cache.get('importances') is not None:
                importances = step._shap_cache['importances']
                param_names = step.input_space.get_hyperparameter_names() if step.input_space else []
                importance_dict = {name: float(imp) for name, imp in zip(param_names, importances)}
                step_info['parameter_importances'] = importance_dict
                step_info['selected_params'] = getattr(step, 'selected_param_names', [])
            
            if hasattr(step, 'compression_info'):
                step_info['compression_details'] = step.compression_info if step.compression_info else {}
            
            steps_info.append(step_info)
        
        info['steps'] = steps_info
    
    with open(save_path, 'w') as f:
        json.dump(info, f, indent=2, default=str)
    
    print(f"Saved compression info to {save_path}")



def visualize_importance_heatmap(param_names: List[str], importances: np.ndarray, 
                                 save_path: str, tasks: Optional[List[str]] = None):
    import seaborn as sns
    
    if len(importances.shape) == 1:
        importances = importances.reshape(1, -1)
    
    n_tasks, n_params = importances.shape
    
    if tasks is None:
        tasks = [f'Task {i+1}' for i in range(n_tasks)]
    
    if n_params > 30:
        mean_importance = importances.mean(axis=0)
        top_indices = np.argsort(mean_importance)[-30:]
        importances = importances[:, top_indices]
        param_names = [param_names[i] for i in top_indices]
        n_params = 30
    
    short_names = [name.split('.')[-1] if len(name) > 20 else name for name in param_names]
    
    fig, ax = plt.subplots(figsize=(max(12, n_params * 0.4), max(6, n_tasks * 0.5)))
    
    sns.heatmap(importances, annot=False, fmt='.3f', cmap='YlOrRd', 
                xticklabels=short_names, yticklabels=tasks,
                cbar_kws={'label': 'Importance Score'}, ax=ax)
    
    ax.set_xlabel('Parameters', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tasks', fontsize=12, fontweight='bold')
    ax.set_title('Parameter Importance Heatmap', fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Saved importance heatmap to {save_path}")
