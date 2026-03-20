import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from ConfigSpace import ConfigurationSpace
import json
import os
from openbox import logger

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def visualize_range_compression_step(step, step_index: int, save_dir: str):    
    if not (hasattr(step, 'compression_info') and step.compression_info):
        return
    
    info = step.compression_info
    if 'compressed_params' not in info or len(info['compressed_params']) == 0:
        return
    
    fig = plt.figure(figsize=(14, max(8, len(info['compressed_params']) * 0.4)))
    
    compressed_params = info['compressed_params']
    n_params = len(compressed_params)
    
    if n_params > 30:
        compressed_params = compressed_params[:30]
        n_params = 30
    
    param_names = [p['name'].split('.')[-1] for p in compressed_params]
    
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
        
        is_quantization = label != ''
        
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
        color = plt.cm.RdYlGn_r(ratio)
        
        if is_quantization:
            ax.barh(idx, norm_comp_end - norm_comp_start, left=norm_comp_start, height=0.4,
                   alpha=0.5, color=color, edgecolor=color, linewidth=2, linestyle='--',
                   label='Quantized (mapped)' if idx == 0 and label else '')
        else:
            ax.barh(idx, norm_comp_end - norm_comp_start, left=norm_comp_start, height=0.4,
                   alpha=0.8, color=color, label='Compressed' if idx == 0 and not label else '')
        
        if label:
            ax.text(1.02, idx, f'{ratio:.1%} ({label})', va='center', fontsize=7)
            ax.text(0.5, idx - 0.35, f'→[{int(comp_min)}, {int(comp_max)}]', 
                   va='top', ha='center', fontsize=12, color='black', 
                   fontweight='bold', style='italic')
        else:
            ax.text(1.02, idx, f'{ratio:.1%}', va='center', fontsize=8)
            ax.text(0.5, idx - 0.35, f'→[{comp_min:.0f}, {comp_max:.0f}]', 
                   va='top', ha='center', fontsize=12, color='black', 
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
    plt.savefig(os.path.join(save_dir, f'range_compression_step_{step_index}.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved range_compression_step_{step_index}.png")


def visualize_compression_summary(pipeline, save_path: str):
    """
    Generate a 4-panel compression summary visualization.
    
    Args:
        pipeline: CompressionPipeline with compression steps
        save_path: Path to save the summary plot
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel 1: Dimension Reduction Across Steps
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
    
    # Panel 2: Compression Ratio by Step
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
    
    # Panel 3: Range Compression Statistics
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
    
    # Panel 4: Text Summary
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
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved compression_summary.png")


def visualize_compression_details(compressor, save_dir: str):
    """
    Intelligently visualize compression details based on the steps used.
    
    This is the main dispatcher that coordinates all visualization functions.
    It automatically generates relevant plots based on:
    - Compression steps used (dimension selection, range compression, projection)
    - Whether transfer learning is used (source similarities)
    - Whether adaptive updates are used
    - Whether multi-task data is available
    
    Automatically generates:
    - Compression summary (always)
    - Range compression details (if range compression steps are used)
    - Parameter importance (if importance-based dimension selection is used)
    - Multi-task importance heatmap (if multiple source tasks are used)
    - Dimension evolution (if adaptive dimension step with history is used)
    - Source task similarities (if transfer learning is used)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    pipeline = compressor.pipeline
    if pipeline is None:
        logger.warning("No pipeline found in compressor")
        return
    
    # 1. Generate compression summary (always)
    visualize_compression_summary(
        pipeline=pipeline,
        save_path=os.path.join(save_dir, 'compression_summary.png')
    )
    
    # 2. Generate range compression details for each step
    for i, step in enumerate(pipeline.steps):
        visualize_range_compression_step(
            step=step,
            step_index=i+1,
            save_dir=save_dir
        )
    
    # Intelligent visualization based on step types
    
    # 1. Generate source task similarity plot if using transfer learning
    if hasattr(compressor, '_source_similarities') and compressor._source_similarities:
        try:
            visualize_source_task_similarities(
                similarities=compressor._source_similarities,
                save_path=os.path.join(save_dir, 'source_task_similarities.png')
            )
        except Exception as e:
            logger.warning(f"Failed to generate source task similarity plot: {e}")
    
    # Check for importance-based dimension selection steps
    for i, step in enumerate(pipeline.steps):
        step_class_name = step.__class__.__name__
        
        # 2. Generate parameter importance plot for SHAP/Correlation/Adaptive steps
        if step_class_name in ['SHAPDimensionStep', 'CorrelationDimensionStep', 'AdaptiveDimensionStep']:
            # Check if importance data is available
            if hasattr(step, '_calculator') and hasattr(step._calculator, '_cache'):
                cache = step._calculator._cache
                if cache and 'importances' in cache and cache['importances'] is not None:
                    importances = cache['importances']
                    if hasattr(step._calculator, 'numeric_hyperparameter_names'):
                        param_names = step._calculator.numeric_hyperparameter_names
                        if len(param_names) > 0 and len(importances) > 0:
                            # Generate single-task importance plot
                            try:
                                visualize_parameter_importance(
                                    param_names=param_names,
                                    importances=importances,
                                    save_path=os.path.join(save_dir, f'parameter_importance_step_{i+1}.png'),
                                    topk=min(20, len(param_names))
                                )
                            except Exception as e:
                                logger.warning(f"Failed to generate parameter importance plot: {e}")
                            
                            # Check if multi-task data is available
                            if (cache.get('importances_per_task') is not None and 
                                cache.get('task_names') is not None):
                                try:
                                    visualize_importance_heatmap(
                                        param_names=param_names,
                                        importances=cache['importances_per_task'],
                                        save_path=os.path.join(save_dir, f'multi_task_importance_heatmap_step_{i+1}.png'),
                                        tasks=cache['task_names']
                                    )
                                except Exception as e:
                                    logger.warning(f"Failed to generate multi-task importance heatmap: {e}")
        
        # 3. Generate dimension evolution plot for Adaptive step
        if step_class_name == 'AdaptiveDimensionStep':
            # Check if we have update history
            if hasattr(compressor, '_dimension_history') and hasattr(compressor, '_iteration_history'):
                iterations = compressor._iteration_history
                dimensions = compressor._dimension_history
                if len(iterations) > 1:  # Has at least initial + 1 update
                    try:
                        visualize_adaptive_dimension_evolution(
                            iterations=iterations,
                            dimensions=dimensions,
                            save_path=os.path.join(save_dir, 'dimension_evolution.png'),
                            title='Adaptive Dimension Evolution'
                        )
                    except Exception as e:
                        logger.warning(f"Failed to generate dimension evolution plot: {e}")


def visualize_parameter_importance(param_names: List[str], importances: List[float], save_path: str, topk: int = 20):
    abs_importances = np.abs(importances)
    sorted_indices = np.argsort(abs_importances)[-topk:]
    top_names = [param_names[i] for i in sorted_indices]
    top_importances = [abs_importances[i] for i in sorted_indices]
    
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


def visualize_adaptive_dimension_evolution(iterations: List[int], dimensions: List[int], 
                                          save_path: str, title: str = 'Adaptive Dimension Evolution'):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(iterations, dimensions, marker='o', linewidth=2, markersize=8, 
           color='steelblue', label='Dimensions')
    
    for i in range(1, len(dimensions)):
        if dimensions[i] != dimensions[i-1]:
            ax.axvline(x=iterations[i], color='red', linestyle='--', alpha=0.5, linewidth=1)
            # Annotate the change
            change = dimensions[i] - dimensions[i-1]
            change_str = f'+{change}' if change > 0 else str(change)
            ax.annotate(f'{dimensions[i]}\n({change_str})', 
                       xy=(iterations[i], dimensions[i]),
                       xytext=(5, 10), textcoords='offset points',
                       fontweight='bold', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Dimensions', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    y_range = max(dimensions) - min(dimensions)
    if y_range > 0:
        ax.set_ylim([min(dimensions) - 0.5, max(dimensions) + 0.5])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved adaptive dimension evolution plot to {save_path}")


def visualize_source_task_similarities(similarities: Dict[int, float], 
                                       save_path: str,
                                       task_names: Optional[List[str]] = None):    
    if not similarities:
        return
    
    task_indices = sorted(similarities.keys())
    sim_values = [similarities[idx] for idx in task_indices]
    
    if task_names is None:
        task_names = [f'Source Task {idx}' for idx in task_indices]
    
    fig, ax = plt.subplots(figsize=(max(10, len(task_indices) * 0.8), 6))
    
    colors = plt.cm.RdYlGn(np.array(sim_values))
    
    bars = ax.bar(range(len(task_indices)), sim_values, color=colors, alpha=0.8, edgecolor='black')
    
    for bar, val in zip(bars, sim_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.3f}',
               ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.set_xlabel('Source Tasks', fontsize=12, fontweight='bold')
    ax.set_ylabel('Similarity Score', fontsize=12, fontweight='bold')
    ax.set_title('Source Task Similarity to Target Task', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(task_indices)))
    ax.set_xticklabels(task_names, rotation=45, ha='right')
    ax.set_ylim([0, max(sim_values) * 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved source_task_similarities.png")


def visualize_importance_heatmap(param_names: List[str], importances: np.ndarray, 
                                 save_path: str, tasks: Optional[List[str]] = None):    
    if len(importances.shape) == 1:
        importances = importances.reshape(1, -1)
    
    importances = np.abs(importances)
    
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
    
    fig, ax = plt.subplots(figsize=(max(14, n_params * 0.5), max(8, n_tasks * 0.6)))
    
    if importances.max() > 0:
        normalized_importances = importances / importances.max()
    else:
        normalized_importances = importances
    
    # Choose colormap (feel free to change):
    # Option 1: 'RdYlGn_r' - Red (low) -> Yellow (medium) -> Green (high) [Intuitive]
    # Option 2: 'viridis' - Purple (low) -> Green -> Yellow (high) [Perceptually uniform]
    # Option 3: 'plasma' - Dark blue (low) -> Purple -> Orange -> Yellow (high) [Good contrast]
    # Option 4: 'rocket_r' - Black (low) -> Red -> Orange (high) [High contrast]
    # Option 5: 'mako_r' - Teal (low) -> Green -> Yellow (high) [Cool tones]
    cmap = sns.color_palette("RdYlGn_r", as_cmap=True)
    
    show_annotations = (n_tasks <= 5 and n_params <= 20)
    
    sns.heatmap(normalized_importances, 
                annot=show_annotations,
                fmt='.2f' if show_annotations else '',
                cmap=cmap,
                xticklabels=short_names, 
                yticklabels=tasks,
                cbar_kws={
                    'label': 'Normalized Importance Score',
                    'orientation': 'vertical',
                    'pad': 0.02
                },
                linewidths=0.5,
                linecolor='white',
                square=False,
                ax=ax)
    
    ax.set_xlabel('Parameters', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel('Tasks', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_title('Multi-Task Parameter Importance Heatmap', 
                fontsize=15, fontweight='bold', pad=15)
    
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=11)
    
    ax.set_facecolor('#f0f0f0')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved multi_task_importance_heatmap.png")
