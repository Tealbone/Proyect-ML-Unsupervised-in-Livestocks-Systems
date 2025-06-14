import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk

class FishClusterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Análisis de Clusters de Especies de Peces")
        self.root.geometry("1200x800")
        
        # Variables
        self.df = None
        self.X_scaled = None
        self.final_clusters = None
        self.optimal_k = 3
        self.kmeans_model = None
        self.scaler = None
        
        # Configurar estilo
        sns.set_style("whitegrid")
        
        # Crear interfaz
        self.create_widgets()
        
    def create_widgets(self):
        # Frame superior para controles
        control_frame = tk.Frame(self.root, padx=10, pady=10)
        control_frame.pack(fill=tk.X)
        
        # Botón para cargar archivo
        tk.Button(
            control_frame, 
            text="Cargar archivo CSV", 
            command=self.load_file,
            bg="#4CAF50",
            fg="white"
        ).pack(side=tk.LEFT, padx=5)
        
        # Selección de k
        tk.Label(control_frame, text="Número de clusters (k):").pack(side=tk.LEFT, padx=5)
        self.k_spinbox = tk.Spinbox(control_frame, from_=2, to=15, width=5)
        self.k_spinbox.pack(side=tk.LEFT, padx=5)
        self.k_spinbox.delete(0, tk.END)
        self.k_spinbox.insert(0, "3")
        
        # Botón para ejecutar análisis
        tk.Button(
            control_frame, 
            text="Ejecutar Análisis", 
            command=self.run_analysis,
            bg="#2196F3",
            fg="white"
        ).pack(side=tk.LEFT, padx=5)
        
        # Botón para mostrar gráficos
        tk.Button(
            control_frame, 
            text="Mostrar Gráficos", 
            command=self.show_plots,
            bg="#FF9800",
            fg="white"
        ).pack(side=tk.LEFT, padx=5)
        
        # Frame principal para resultados
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Notebook para organizar pestañas
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Pestañas
        self.tab1 = ttk.Frame(self.notebook)
        self.tab2 = ttk.Frame(self.notebook)
        self.tab3 = ttk.Frame(self.notebook)
        self.tab4 = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab1, text="Datos")
        self.notebook.add(self.tab2, text="Resultados")
        self.notebook.add(self.tab3, text="Gráficos")
        self.notebook.add(self.tab4, text="Predecir Nuevo Pez")
        
        # Widgets para la pestaña de datos
        self.data_text = tk.Text(self.tab1, wrap=tk.NONE)
        scroll_y = tk.Scrollbar(self.tab1, orient="vertical", command=self.data_text.yview)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.data_text.configure(yscrollcommand=scroll_y.set)
        self.data_text.pack(fill=tk.BOTH, expand=True)
        
        # Widgets para la pestaña de resultados
        self.result_text = tk.Text(self.tab2, wrap=tk.NONE)
        scroll_y_res = tk.Scrollbar(self.tab2, orient="vertical", command=self.result_text.yview)
        scroll_y_res.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.result_text.configure(yscrollcommand=scroll_y_res.set)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
        # Widgets para la pestaña de gráficos
        self.graph_frame = tk.Frame(self.tab3)
        self.graph_frame.pack(fill=tk.BOTH, expand=True)
        
        # Widgets para la pestaña de nuevo pez
        self.create_prediction_tab()
    def create_prediction_tab(self):
        # Frame para entrada de datos
        input_frame = tk.Frame(self.tab4, padx=10, pady=10)
        input_frame.pack(fill=tk.X)
        
        # Campos de entrada
        tk.Label(input_frame, text="Longitud (cm):").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.length_entry = tk.Entry(input_frame)
        self.length_entry.grid(row=0, column=1, padx=5, pady=5)
        
        tk.Label(input_frame, text="Peso (kg):").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.weight_entry = tk.Entry(input_frame)
        self.weight_entry.grid(row=1, column=1, padx=5, pady=5)
        
        tk.Label(input_frame, text="Relación Peso/Long:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.ratio_entry = tk.Entry(input_frame)
        self.ratio_entry.grid(row=2, column=1, padx=5, pady=5)
        
        # Botón de predicción
        predict_btn = tk.Button(
            input_frame, 
            text="Predecir Cluster", 
            command=self.predict_new_fish,
            bg="#9C27B0",
            fg="white"
        )
        predict_btn.grid(row=3, column=0, columnspan=2, pady=10)
        
        # Frame para resultados de predicción
        result_frame = tk.Frame(self.tab4)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Gráfico de predicción
        self.prediction_graph_frame = tk.Frame(result_frame)
        self.prediction_graph_frame.pack(fill=tk.BOTH, expand=True)
        
        # Resultado de predicción
        self.prediction_result = tk.Label(
            result_frame, 
            text="Ingrese los datos del pez y haga clic en 'Predecir Cluster'",
            font=('Helvetica', 12),
            pady=20
        )
        self.prediction_result.pack(fill=tk.X)
    
    def predict_new_fish(self):
        if self.kmeans_model is None or self.scaler is None:
            messagebox.showerror("Error", "Primero debe ejecutar el análisis de clusters")
            return
            
        try:
            # Obtener valores del formulario
            length = float(self.length_entry.get())
            weight = float(self.weight_entry.get())
            ratio = float(self.ratio_entry.get())
            
            # Crear array con los datos del nuevo pez
            new_fish = np.array([[length, weight, ratio]])
            
            # Escalar los datos (usando el mismo scaler que para el modelo)
            new_fish_scaled = self.scaler.transform(new_fish)
            
            # Predecir el cluster
            cluster = self.kmeans_model.predict(new_fish_scaled)[0]
            
            # Mostrar resultado
            self.show_prediction_result(new_fish, cluster)
            
        except ValueError as e:
            messagebox.showerror("Error", f"Datos inválidos: {str(e)}")
    
    def show_prediction_result(self, new_fish, cluster):
        # Limpiar frame de gráfico
        for widget in self.prediction_graph_frame.winfo_children():
            widget.destroy()
            
        # Crear figura con los clusters y el nuevo punto
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        
        # Graficar clusters existentes
        if self.final_clusters is not None:
            sns.scatterplot(
                data=self.final_clusters, 
                x='length', 
                y='weight', 
                hue='cluster', 
                palette='viridis', 
                s=80,
                alpha=0.7,
                ax=ax
            )
        
        # Graficar nuevo punto (en rojo para destacar)
        ax.scatter(
            new_fish[0, 0], 
            new_fish[0, 1], 
            c='red', 
            s=150, 
            marker='X', 
            label='Nuevo pez',
            edgecolors='black'
        )
        
        ax.set_title('Posición del Nuevo Pez en los Clusters')
        ax.set_xlabel('Longitud (cm)')
        ax.set_ylabel('Peso (kg)')
        ax.legend()
        
        # Integrar figura en Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.prediction_graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Mostrar descripción del cluster
        cluster_desc = self.get_cluster_description(cluster)
        self.prediction_result.config(
            text=f"Este pez pertenece al Cluster {cluster}\n{cluster_desc}",
            fg="#4CAF50"
        )
    
    def get_cluster_description(self, cluster):
        if self.final_clusters is None:
            return ""
            
        # Obtener estadísticas del cluster
        cluster_data = self.final_clusters[self.final_clusters['cluster'] == cluster]
        
        # Calcular promedios
        avg_length = cluster_data['length'].mean()
        avg_weight = cluster_data['weight'].mean()
        avg_ratio = cluster_data['w_l_ratio'].mean()
        
        # Determinar tipo de pez
        if avg_length < 15 and avg_weight < 5:
            pez_type = "pequeños con crecimiento rápido"
        elif avg_length < 25:
            pez_type = "medianos en desarrollo"
        else:
            pez_type = "grandes maduros"
            
        return (
            f"Características típicas:\n"
            f"- Longitud promedio: {avg_length:.1f} cm\n"
            f"- Peso promedio: {avg_weight:.2f} kg\n"
            f"- Relación peso/longitud: {avg_ratio:.2f}\n"
            f"Este cluster contiene peces {pez_type}."
        )

    def load_file(self):
        file_path = filedialog.askopenfilename(
            title="Seleccionar archivo CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.data_text.delete(1.0, tk.END)
                self.data_text.insert(tk.END, "Información del dataset:\n")
                self.data_text.insert(tk.END, f"{self.df.info()}\n\n")
                self.data_text.insert(tk.END, "Estadísticas descriptivas:\n")
                self.data_text.insert(tk.END, f"{self.df.describe().to_string()}")
                
                # Preprocesamiento automático
                self.preprocess_data()
                
            except Exception as e:
                self.data_text.delete(1.0, tk.END)
                self.data_text.insert(tk.END, f"Error al cargar el archivo: {str(e)}")
    
    def preprocess_data(self):
        if self.df is not None:
            try:
                # Seleccionar características
                X = self.df[['length', 'weight', 'w_l_ratio']]
                
                # Escalar datos
                scaler = StandardScaler()
                self.X_scaled = scaler.fit_transform(X)
                
                self.data_text.insert(tk.END, "\n\nPreprocesamiento completado exitosamente!")
                
            except Exception as e:
                self.data_text.insert(tk.END, f"\nError en preprocesamiento: {str(e)}")
    
    def run_analysis(self):
        if self.X_scaled is None:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Error: Primero debe cargar y preprocesar los datos.")
            return
            
        try:
            k = int(self.k_spinbox.get())
            if k < 2 or k > 15:
                raise ValueError("k debe estar entre 2 y 15")
                
            self.optimal_k = k

            # Guardar el scaler para usarlo en predicciones
            self.scaler = StandardScaler().fit(self.df[['length', 'weight', 'w_l_ratio']])
            
            # Aplicar K-Means y guardar el modelo
            self.kmeans_model = KMeans(n_clusters=k, random_state=42)
            clusters = self.kmeans_model.fit_predict(self.X_scaled)
            
            # Añadir etiquetas al dataframe
            self.final_clusters = self.df.copy()
            self.final_clusters['cluster'] = clusters
            
            # Mostrar resultados
            self.show_results()
            
        except Exception as e:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Error en el análisis: {str(e)}")
    
    def cluster_with_k(self, k, data_scaled, original_data):
        # Aplicar K-Means
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(data_scaled)
        
        # Añadir etiquetas al dataframe
        clustered_data = original_data.copy()
        clustered_data['cluster'] = clusters
        
        return clustered_data
    
    def show_results(self):
        if self.final_clusters is None:
            return
            
        self.result_text.delete(1.0, tk.END)
        
        # Obtener centroides
        kmeans_final = KMeans(n_clusters=self.optimal_k, random_state=42).fit(self.X_scaled)
        centroids_scaled = kmeans_final.cluster_centers_
        scaler = StandardScaler().fit(self.df[['length', 'weight', 'w_l_ratio']])
        centroids_original = scaler.inverse_transform(centroids_scaled)
        centroids_df = pd.DataFrame(centroids_original, columns=['length', 'weight', 'w_l_ratio'])
        centroids_df['cluster'] = centroids_df.index
        
        # Mostrar centroides
        self.result_text.insert(tk.END, "=== REPORTE DE CLUSTERS ===\n\n")
        self.result_text.insert(tk.END, "Características medias de cada cluster:\n")
        self.result_text.insert(tk.END, f"{centroids_df.to_string()}\n\n")
        
        # Interpretación de clusters
        for i in range(self.optimal_k):
            c = centroids_df.iloc[i]
            self.result_text.insert(tk.END, f"Cluster {i}:\n")
            self.result_text.insert(tk.END, f"- Longitud promedio: {c['length']:.1f} cm\n")
            self.result_text.insert(tk.END, f"- Peso promedio: {c['weight']:.2f} kg\n")
            self.result_text.insert(tk.END, f"- Relación peso/longitud: {c['w_l_ratio']:.2f}\n\n")
        
        # Distribución de especies
        if 'species' in self.final_clusters.columns:
            self.result_text.insert(tk.END, "\nDistribución de especies por cluster:\n")
            species_cluster = pd.crosstab(self.final_clusters['species'], self.final_clusters['cluster'])
            self.result_text.insert(tk.END, f"{species_cluster.to_string()}\n")
    
    def show_plots(self):
        if self.final_clusters is None:
            return
            
        # Limpiar frame de gráficos
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
            
        # Crear figura
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Gráfico 1: Longitud vs Peso
        sns.scatterplot(
            data=self.final_clusters, 
            x='length', 
            y='weight', 
            hue='cluster', 
            palette='viridis', 
            s=100, 
            ax=axes[0]
        )
        axes[0].set_title('Longitud vs Peso por Cluster')
        axes[0].set_xlabel('Longitud (cm)')
        axes[0].set_ylabel('Peso (kg)')
        
        # Gráfico 2: Relación peso/longitud
        sns.boxplot(
            data=self.final_clusters, 
            x='cluster', 
            y='w_l_ratio', 
            palette='viridis', 
            ax=axes[1]
        )
        axes[1].set_title('Relación Peso/Longitud por Cluster')
        axes[1].set_xlabel('Cluster')
        axes[1].set_ylabel('Relación Peso/Longitud')
        
        plt.tight_layout()
        
        # Integrar figura en Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Gráfico de métricas (Elbow y Silhouette)
        self.show_metrics_plots()
    
    def show_metrics_plots(self):
        # Frame adicional para métricas
        metrics_frame = tk.Frame(self.graph_frame)
        metrics_frame.pack(fill=tk.BOTH, expand=True)
        
        # Calcular métricas
        inertias = []
        silhouette_scores = []
        max_k = min(10, len(self.X_scaled)-1)
        
        for k in range(1, max_k+1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.X_scaled)
            inertias.append(kmeans.inertia_)
            
            if k > 1:
                cluster_labels = kmeans.predict(self.X_scaled)
                silhouette_scores.append(silhouette_score(self.X_scaled, cluster_labels))
        
        # Crear figura de métricas
        fig_metrics, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Gráfico Elbow
        ax1.plot(range(1, max_k+1), inertias, marker='o')
        ax1.set_title('Método del Codo')
        ax1.set_xlabel('Número de Clusters (k)')
        ax1.set_ylabel('Inercia (SSE)')
        ax1.axvline(self.optimal_k, color='r', linestyle='--')
        ax1.grid(True)
        
        # Gráfico Silhouette
        if silhouette_scores:
            ax2.plot(range(2, max_k+1), silhouette_scores, marker='o')
            ax2.set_title('Coeficiente de Silueta')
            ax2.set_xlabel('Número de Clusters (k)')
            ax2.set_ylabel('Coeficiente de Silueta')
            ax2.axvline(self.optimal_k, color='r', linestyle='--')
            ax2.grid(True)
        
        plt.tight_layout()
        
        # Integrar figura en Tkinter
        canvas_metrics = FigureCanvasTkAgg(fig_metrics, master=metrics_frame)
        canvas_metrics.draw()
        canvas_metrics.get_tk_widget().pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = FishClusterApp(root)
    root.mainloop()