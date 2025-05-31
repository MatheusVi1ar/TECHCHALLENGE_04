from matplotlib import pyplot as plt


def plot_metrics_comparison(train_metrics, test_metrics):
    """Plota comparação das métricas entre treino, validação e teste"""
    metrics_names = list(train_metrics.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, metric in enumerate(metrics_names):
        values = [train_metrics[metric], test_metrics[metric]]
        labels = ['Treino', 'Teste']
        colors = ['#2E86AB', '#F18F01']
        
        bars = axes[i].bar(labels, values, color=colors, alpha=0.8, edgecolor='black')
        axes[i].set_title(f'{metric}', fontsize=14, fontweight='bold')
        axes[i].set_ylabel('Valor')
        axes[i].grid(True, alpha=0.3, axis='y')
        
        # Adiciona valores nas barras
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Comparação de Métricas - Treino vs Teste', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_predictions_vs_actual(dates, actual, predicted, title, subset_size=200):
    """Plota valores observados vs previstos"""
    # Se temos muitos dados, mostra apenas uma amostra para visualização
    if len(actual) > subset_size:
        step = len(actual) // subset_size
        dates_plot = dates[::step]
        actual_plot = actual[::step]
        predicted_plot = predicted[::step]
    else:
        dates_plot = dates
        actual_plot = actual
        predicted_plot = predicted
    
    plt.figure(figsize=(15, 8))
    
    plt.plot(dates_plot, actual_plot, label='Valores Reais', 
             color='#2E86AB', linewidth=2, alpha=0.8)
    plt.plot(dates_plot, predicted_plot, label='Previsões', 
             color='#A23B72', linewidth=2, alpha=0.8)
    
    plt.title(f'{title}', fontsize=16, fontweight='bold')
    plt.xlabel('Data', fontsize=12)
    plt.ylabel('Preço ($)', fontsize=12)
    plt.legend(fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_training_progress(model):
    """Plota o progresso do treinamento"""
    fig, (ax1) = plt.subplots(1, figsize=(15, 5))
    
    # Loss de treino
    if model.training_losses:
        ax1.plot(model.training_losses, color='#2E86AB', linewidth=2)
        ax1.set_title('Loss de Treinamento', fontweight='bold')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('MSE Loss')
        ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()