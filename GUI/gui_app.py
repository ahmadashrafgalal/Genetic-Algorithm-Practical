import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import numpy as np
import networkx as nx
from sklearn.datasets import make_friedman1
from sklearn.linear_model import LinearRegression

from Algorithms.KnapSack import Knapsack
from Algorithms.TSP import TSP
from Algorithms.NQueen import NQueens
from Algorithms.GraphColoring import GraphColoring
from Algorithms.PSO import PSO
from Algorithms.VRP import VRP
from Algorithms.NurseScheduling import NurseScheduling
from Algorithms.FeatureSelection import FeatureSelection


class AlgorithmGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Genetic Algorithm Visualization Suite")
        self.root.geometry("1400x900")
        self.root.configure(bg="#2b2b2b")
        
        self.setup_styles()
        
        self.main_paned = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.left_frame = ttk.Frame(self.main_paned, width=350)
        self.main_paned.add(self.left_frame, weight=0)
        self.setup_left_panel()
        
        self.right_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.right_frame, weight=1)
        self.setup_right_panel()
        
        self.running = False
        self.algorithm_instance = None
        
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TLabel', background='#f0f0f0', foreground='#333')
        style.configure('TButton', background='#007ACC')
        style.configure('Title.TLabel', font=('Arial', 14, 'bold'), background='#f0f0f0')
        style.configure('Section.TLabel', font=('Arial', 11, 'bold'), background='#f0f0f0')
        
    def setup_left_panel(self):
        title_label = ttk.Label(self.left_frame, text="Algorithm Selector", style='Title.TLabel')
        title_label.pack(pady=10, padx=10)
        
        algorithms = [
            "Knapsack Problem",
            "Traveling Salesman Problem (TSP)",
            "N-Queens Problem",
            "Graph Coloring",
            "Particle Swarm Optimization (PSO)",
            "Vehicle Routing Problem (VRP)",
            "Nurse Scheduling",
            "Feature Selection"
        ]
        
        self.selected_algo = tk.StringVar(value=algorithms[0])
        self.algo_combo = ttk.Combobox(
            self.left_frame, 
            textvariable=self.selected_algo,
            values=algorithms,
            state='readonly',
            width=40
        )
        self.algo_combo.pack(pady=10, padx=10)
        self.algo_combo.bind('<<ComboboxSelected>>', self.on_algorithm_changed)
        
        params_label = ttk.Label(self.left_frame, text="Parameters", style='Section.TLabel')
        params_label.pack(pady=(20, 10), padx=10)
        
        # Scrollable frame for parameters
        self.params_canvas = tk.Canvas(self.left_frame, bg='#f0f0f0', highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.left_frame, orient=tk.VERTICAL, command=self.params_canvas.yview)
        self.params_scrollable_frame = ttk.Frame(self.params_canvas)
        
        self.params_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.params_canvas.configure(scrollregion=self.params_canvas.bbox("all"))
        )
        
        self.params_canvas.create_window((0, 0), window=self.params_scrollable_frame, anchor="nw")
        self.params_canvas.configure(yscrollcommand=scrollbar.set)
        
        self.params_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        button_frame = ttk.Frame(self.left_frame)
        button_frame.pack(pady=20, padx=10, fill=tk.X)
        
        self.run_button = ttk.Button(button_frame, text="▶ Run Algorithm", command=self.run_algorithm)
        self.run_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="⏹ Stop", command=self.stop_algorithm, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        results_label = ttk.Label(self.left_frame, text="Results", style='Section.TLabel')
        results_label.pack(pady=(20, 10), padx=10)
        
        self.results_text = tk.Text(self.left_frame, height=12, width=40, bg='#fff', fg='#000', relief=tk.SUNKEN)
        self.results_text.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        
        self.on_algorithm_changed(None)
        
    def setup_right_panel(self):
        self.viz_frame = ttk.Frame(self.right_frame)
        self.viz_frame.pack(fill=tk.BOTH, expand=True)
        
        self.fig = Figure(figsize=(10, 8), dpi=100, facecolor='#f0f0f0')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.info_label = ttk.Label(self.right_frame, text="Select an algorithm and click 'Run' to visualize", 
                                     style='Section.TLabel', foreground='#666')
        self.info_label.pack(pady=10)
        
    def on_algorithm_changed(self, event):
        for widget in self.params_scrollable_frame.winfo_children():
            widget.destroy()
        
        algo = self.selected_algo.get()
        self.current_params = {}
        
        if "Knapsack" in algo:
            self.create_knapsack_params()
        elif "TSP" in algo:
            self.create_tsp_params()
        elif "N-Queens" in algo:
            self.create_nqueens_params()
        elif "Graph Coloring" in algo:
            self.create_graph_coloring_params()
        elif "Particle Swarm" in algo:
            self.create_pso_params()
        elif "Vehicle Routing" in algo:
            self.create_vrp_params()
        elif "Nurse Scheduling" in algo:
            self.create_nurse_scheduling_params()
        elif "Feature Selection" in algo:
            self.create_feature_selection_params()
    
    def create_param_entry(self, label, default_value, param_type="entry"):
        frame = ttk.Frame(self.params_scrollable_frame)
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        label_widget = ttk.Label(frame, text=label)
        label_widget.pack(anchor=tk.W)
        
        if param_type == "entry":
            entry = ttk.Entry(frame, width=30)
            entry.insert(0, str(default_value))
            entry.pack(fill=tk.X, pady=2)
            self.current_params[label] = entry
        elif param_type == "spinbox":
            spinbox = ttk.Spinbox(frame, from_=1, to=1000, width=30)
            spinbox.set(default_value)
            spinbox.pack(fill=tk.X, pady=2)
            self.current_params[label] = spinbox
        
        return frame
    
    def create_knapsack_params(self):
        ttk.Label(self.params_scrollable_frame, text="Knapsack Problem", style='Section.TLabel').pack(pady=5)
        
        frame = ttk.Frame(self.params_scrollable_frame)
        frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(frame, text="Items (comma-separated):").pack(anchor=tk.W)
        items_entry = ttk.Entry(frame, width=30)
        items_entry.insert(0, "A,B,C,D,E")
        items_entry.pack(fill=tk.X, pady=2)
        self.current_params["Items"] = items_entry
        
        frame = ttk.Frame(self.params_scrollable_frame)
        frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(frame, text="Values (comma-separated):").pack(anchor=tk.W)
        values_entry = ttk.Entry(frame, width=30)
        values_entry.insert(0, "50,30,20,30,50")
        values_entry.pack(fill=tk.X, pady=2)
        self.current_params["Values"] = values_entry
        
        frame = ttk.Frame(self.params_scrollable_frame)
        frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(frame, text="Weights (comma-separated):").pack(anchor=tk.W)
        weights_entry = ttk.Entry(frame, width=30)
        weights_entry.insert(0, "30,50,40,20,60")
        weights_entry.pack(fill=tk.X, pady=2)
        self.current_params["Weights"] = weights_entry
        
        self.create_param_entry("Capacity:", 110, "spinbox")
        self.create_param_entry("Population Size:", 6, "spinbox")
        self.create_param_entry("Generations:", 30, "spinbox")
        self.create_param_entry("Mutation Rate:", 0.1, "entry")
    
    def create_tsp_params(self):
        ttk.Label(self.params_scrollable_frame, text="TSP Problem", style='Section.TLabel').pack(pady=5)
        
        frame = ttk.Frame(self.params_scrollable_frame)
        frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(frame, text="Number of Cities:").pack(anchor=tk.W)
        cities_spinbox = ttk.Spinbox(frame, from_=3, to=50, width=30)
        cities_spinbox.set(5)
        cities_spinbox.pack(fill=tk.X, pady=2)
        self.current_params["Num Cities"] = cities_spinbox
        
        self.create_param_entry("Population Size:", 30, "spinbox")
        self.create_param_entry("Generations:", 40, "spinbox")
        self.create_param_entry("Mutation Rate:", 0.25, "entry")
    
    def create_nqueens_params(self):
        ttk.Label(self.params_scrollable_frame, text="N-Queens Problem", style='Section.TLabel').pack(pady=5)
        
        self.create_param_entry("Board Size (N):", 8, "spinbox")
        self.create_param_entry("Population Size:", 100, "spinbox")
        self.create_param_entry("Generations:", 200, "spinbox")
        self.create_param_entry("Mutation Rate:", 0.01, "entry")
    
    def create_graph_coloring_params(self):
        ttk.Label(self.params_scrollable_frame, text="Graph Coloring", style='Section.TLabel').pack(pady=5)
        
        self.create_param_entry("Number of Nodes:", 6, "spinbox")
        self.create_param_entry("Number of Colors:", 3, "spinbox")
        self.create_param_entry("Population Size:", 20, "spinbox")
        self.create_param_entry("Generations:", 50, "spinbox")
        self.create_param_entry("Mutation Rate:", 0.3, "entry")
    
    def create_pso_params(self):
        ttk.Label(self.params_scrollable_frame, text="PSO", style='Section.TLabel').pack(pady=5)
        
        self.create_param_entry("Dimensions:", 2, "spinbox")
        self.create_param_entry("Population:", 30, "spinbox")
        self.create_param_entry("Max Iterations:", 100, "spinbox")
        self.create_param_entry("Min Bound:", -5, "entry")
        self.create_param_entry("Max Bound:", 5, "entry")
        self.create_param_entry("Inertia (w):", 0.7, "entry")
        self.create_param_entry("Cognitive (c1):", 1.5, "entry")
        self.create_param_entry("Social (c2):", 1.5, "entry")
    
    def create_vrp_params(self):
        ttk.Label(self.params_scrollable_frame, text="VRP", style='Section.TLabel').pack(pady=5)
        
        self.create_param_entry("Number of Vehicles:", 2, "spinbox")
        self.create_param_entry("Number of Customers:", 5, "spinbox")
        self.create_param_entry("Population Size:", 20, "spinbox")
        self.create_param_entry("Generations:", 50, "spinbox")
        self.create_param_entry("Mutation Rate:", 0.1, "entry")
    
    def create_nurse_scheduling_params(self):
        ttk.Label(self.params_scrollable_frame, text="Nurse Scheduling", style='Section.TLabel').pack(pady=5)
        
        self.create_param_entry("Number of Nurses:", 4, "spinbox")
        self.create_param_entry("Number of Days:", 5, "spinbox")
        self.create_param_entry("Max Shifts per Nurse:", 4, "spinbox")
        self.create_param_entry("Population Size:", 20, "spinbox")
        self.create_param_entry("Generations:", 50, "spinbox")
        self.create_param_entry("Mutation Rate:", 0.3, "entry")
    
    def create_feature_selection_params(self):
        ttk.Label(self.params_scrollable_frame, text="Feature Selection", style='Section.TLabel').pack(pady=5)
        
        self.create_param_entry("Number of Samples:", 100, "spinbox")
        self.create_param_entry("Number of Features:", 10, "spinbox")
        self.create_param_entry("Population Size:", 20, "spinbox")
        self.create_param_entry("Generations:", 30, "spinbox")
        self.create_param_entry("Mutation Rate:", 0.3, "entry")
    
    def get_param_value(self, label, param_type="str"):
        widget = self.current_params.get(label)
        if widget:
            value = widget.get()
            if param_type == "int":
                return int(value)
            elif param_type == "float":
                return float(value)
            return value
        return None
    
    def run_algorithm(self):
        if self.running:
            messagebox.showwarning("Running", "Algorithm is already running!")
            return
        
        self.running = True
        self.run_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        thread = threading.Thread(target=self.execute_algorithm)
        thread.daemon = True
        thread.start()
    
    def stop_algorithm(self):
        self.running = False
        self.run_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.results_text.insert(tk.END, "\nAlgorithm stopped by user.")
    
    def execute_algorithm(self):
        try:
            algo = self.selected_algo.get()
            
            if "Knapsack" in algo:
                self.run_knapsack()
            elif "TSP" in algo:
                self.run_tsp()
            elif "N-Queens" in algo:
                self.run_nqueens()
            elif "Graph Coloring" in algo:
                self.run_graph_coloring()
            elif "Particle Swarm" in algo:
                self.run_pso()
            elif "Vehicle Routing" in algo:
                self.run_vrp()
            elif "Nurse Scheduling" in algo:
                self.run_nurse_scheduling()
            elif "Feature Selection" in algo:
                self.run_feature_selection()
        except Exception as e:
            self.results_text.insert(tk.END, f"Error: {str(e)}")
        finally:
            self.running = False
            self.run_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
    
    def run_knapsack(self):
        try:
            items = self.get_param_value("Items").split(',')
            values = list(map(int, self.get_param_value("Values").split(',')))
            weights = list(map(int, self.get_param_value("Weights").split(',')))
            capacity = self.get_param_value("Capacity:", "int")
            pop_size = self.get_param_value("Population Size:", "int")
            generations = self.get_param_value("Generations:", "int")
            mutation_rate = self.get_param_value("Mutation Rate:", "float")
            
            ga = Knapsack(
                items=items,
                values=values,
                weights=weights,
                capacity=capacity,
                population_size=pop_size,
                generations=generations,
                mutation_rate=mutation_rate
            )
            
            self.results_text.insert(tk.END, "Running Knapsack GA...\n")
            self.root.update()
            
            best_solution = ga.run()
            
            total_value = sum([values[i] for i in range(len(items)) if best_solution[i] == 1])
            total_weight = sum([weights[i] for i in range(len(items)) if best_solution[i] == 1])
            selected_items = [items[i] for i in range(len(items)) if best_solution[i] == 1]
            
            results = f"""
Knapsack Problem Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━
Selected Items: {', '.join(selected_items)}
Total Value: {total_value}
Total Weight: {total_weight}
Capacity: {capacity}
Space Used: {total_weight}/{capacity}
Efficiency: {(total_value/capacity)*100:.2f}%
"""
            self.results_text.insert(tk.END, results)
            
            # Visualize
            self.visualize_knapsack(items, values, weights, best_solution, capacity)
        
        except Exception as e:
            self.results_text.insert(tk.END, f"Error: {str(e)}\n")
    
    def visualize_knapsack(self, items, values, weights, solution, capacity):
        self.fig.clear()
        ax1 = self.fig.add_subplot(121)
        ax2 = self.fig.add_subplot(122)
        
        selected_items = [items[i] for i in range(len(items)) if solution[i] == 1]
        selected_values = [values[i] for i in range(len(items)) if solution[i] == 1]
        
        colors = ['#4CAF50' if solution[i] == 1 else '#ccc' for i in range(len(items))]
        ax1.bar(items, values, color=colors)
        ax1.set_xlabel('Items')
        ax1.set_ylabel('Value')
        ax1.set_title('Item Values (Green = Selected)')
        ax1.grid(axis='y', alpha=0.3)
        
        selected_weight = sum([weights[i] for i in range(len(items)) if solution[i] == 1])
        remaining = capacity - selected_weight
        
        ax2.pie([selected_weight, remaining], labels=['Used', 'Remaining'], 
                autopct='%1.1f%%', colors=['#FF9800', '#ddd'])
        ax2.set_title(f'Knapsack Capacity\n({selected_weight}/{capacity})')
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def run_tsp(self):
        try:
            num_cities = self.get_param_value("Num Cities", "int")
            pop_size = self.get_param_value("Population Size:", "int")
            generations = self.get_param_value("Generations:", "int")
            mutation_rate = self.get_param_value("Mutation Rate:", "float")
            
            np.random.seed(42)
            cities_coords = np.random.rand(num_cities, 2) * 10
            city_labels = [f"C{i}" for i in range(num_cities)]
            
            tsp = TSP(
                cities_coords=cities_coords,
                city_labels=city_labels,
                population_size=pop_size,
                generations=generations,
                mutation_rate=mutation_rate
            )
            
            self.results_text.insert(tk.END, "Running TSP GA...\n")
            self.root.update()
            
            best_route = tsp.run()
            best_cost = tsp.best_cost_overall
            route_string = " -> ".join([city_labels[i] for i in best_route]) + f" -> {city_labels[best_route[0]]}"
            
            results = f"""
TSP Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━
Best Route: {route_string}
Total Distance: {best_cost:.2f}
Number of Cities: {num_cities}
Average City Distance: {best_cost/num_cities:.2f}
"""
            self.results_text.insert(tk.END, results)
            
            self.visualize_tsp(cities_coords, city_labels, best_route, best_cost)
        
        except Exception as e:
            self.results_text.insert(tk.END, f"Error: {str(e)}\n")
    
    def visualize_tsp(self, cities_coords, city_labels, best_route, best_cost):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        ax.scatter(cities_coords[:, 0], cities_coords[:, 1], s=200, c='#FF6B6B', zorder=3)
        
        for i, label in enumerate(city_labels):
            ax.annotate(label, (cities_coords[i, 0], cities_coords[i, 1]), 
                       fontsize=10, fontweight='bold', ha='center', va='center')
        
        for i in range(len(best_route)):
            start = best_route[i]
            end = best_route[(i + 1) % len(best_route)]
            ax.arrow(cities_coords[start, 0], cities_coords[start, 1],
                    cities_coords[end, 0] - cities_coords[start, 0],
                    cities_coords[end, 1] - cities_coords[start, 1],
                    head_width=0.2, head_length=0.1, fc='#2196F3', ec='#2196F3', alpha=0.7, length_includes_head=True)
        
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title(f'TSP Solution - Distance: {best_cost:.2f}')
        ax.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def run_nqueens(self):
        try:
            n = self.get_param_value("Board Size (N):", "int")
            pop_size = self.get_param_value("Population Size:", "int")
            generations = self.get_param_value("Generations:", "int")
            mutation_rate = self.get_param_value("Mutation Rate:", "float")
            
            ga = NQueens(
                n=n,
                population_size=pop_size,
                generations=generations,
                mutation_rate=mutation_rate
            )
            
            self.results_text.insert(tk.END, "Running N-Queens GA...\n")
            self.root.update()
            
            best_solution = ga.run()
            
            # Count conflicts
            conflicts = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if abs(best_solution[i] - best_solution[j]) == abs(i - j):
                        conflicts += 1
            
            results = f"""
N-Queens Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━
Board Size: {n}x{n}
Solution: {best_solution}
Conflicts: {conflicts}
Status: {'✓ SOLVED' if conflicts == 0 else '⚠ Partial Solution'}
"""
            self.results_text.insert(tk.END, results)
            
            self.visualize_nqueens(n, best_solution)
        
        except Exception as e:
            self.results_text.insert(tk.END, f"Error: {str(e)}\n")
    
    def visualize_nqueens(self, n, solution):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        for i in range(n):
            for j in range(n):
                color = '#fff' if (i + j) % 2 == 0 else '#d0d0d0'
                rect = plt.Rectangle((j, n-1-i), 1, 1, facecolor=color, edgecolor='black')
                ax.add_patch(rect)
        
        for col, row in enumerate(solution):
            ax.plot(col + 0.5, n - 1 - row + 0.5, 'r*', markersize=25)
        
        ax.set_xlim(0, n)
        ax.set_ylim(0, n)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'N-Queens Solution (Board Size: {n})')
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def run_graph_coloring(self):
        try:
            num_nodes = self.get_param_value("Number of Nodes:", "int")
            num_colors = self.get_param_value("Number of Colors:", "int")
            pop_size = self.get_param_value("Population Size:", "int")
            generations = self.get_param_value("Generations:", "int")
            mutation_rate = self.get_param_value("Mutation Rate:", "float")
            
            G = nx.erdos_renyi_graph(num_nodes, 0.4, seed=42)
            
            ga = GraphColoring(
                graph=G,
                num_colors=num_colors,
                population_size=pop_size,
                generations=generations,
                mutation_rate=mutation_rate
            )
            
            self.results_text.insert(tk.END, "Running Graph Coloring GA...\n")
            self.root.update()
            
            best_coloring = ga.run()
            
            conflicts = 0
            for u, v in G.edges():
                if best_coloring[u] == best_coloring[v]:
                    conflicts += 1
            
            results = f"""
Graph Coloring Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━
Number of Nodes: {num_nodes}
Number of Colors: {num_colors}
Conflicts: {conflicts}
Coloring: {best_coloring}
Status: {'✓ Valid Coloring' if conflicts == 0 else f'⚠ {conflicts} conflicts'}
"""
            self.results_text.insert(tk.END, results)
            
            self.visualize_graph_coloring(G, best_coloring)
        
        except Exception as e:
            self.results_text.insert(tk.END, f"Error: {str(e)}\n")
    
    def visualize_graph_coloring(self, G, coloring):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        pos = nx.spring_layout(G, seed=42)
        
        colors_map = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
        node_colors = [colors_map[coloring[node] % len(colors_map)] for node in G.nodes()]
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, ax=ax)
        nx.draw_networkx_edges(G, pos, ax=ax, width=2, alpha=0.6)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
        
        ax.set_title('Graph Coloring Solution')
        ax.axis('off')
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def run_pso(self):
        try:
            dimensions = self.get_param_value("Dimensions:", "int")
            population = self.get_param_value("Population:", "int")
            max_iter = self.get_param_value("Max Iterations:", "int")
            min_bound = self.get_param_value("Min Bound:", "float")
            max_bound = self.get_param_value("Max Bound:", "float")
            w = self.get_param_value("Inertia (w):", "float")
            c1 = self.get_param_value("Cognitive (c1):", "float")
            c2 = self.get_param_value("Social (c2):", "float")
            
            pso = PSO(
                dimensions=dimensions,
                population=population,
                max_iter=max_iter,
                min_bound=min_bound,
                max_bound=max_bound,
                w=w,
                c1=c1,
                c2=c2
            )
            
            self.results_text.insert(tk.END, "Running PSO...\n")
            self.root.update()
            
            best_position, best_cost = pso.run()
            
            results = f"""
PSO Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━
Best Position: [{', '.join([f'{x:.4f}' for x in best_position])}]
Best Cost: {best_cost:.6f}
Dimensions: {dimensions}
Population: {population}
Iterations: {max_iter}
"""
            self.results_text.insert(tk.END, results)
            
            self.visualize_pso(pso, dimensions, best_position)
        
        except Exception as e:
            self.results_text.insert(tk.END, f"Error: {str(e)}\n")
    
    def visualize_pso(self, pso, dimensions, best_position):
        self.fig.clear()
        
        if dimensions == 2:
            # 2D visualization
            ax = self.fig.add_subplot(111)
            
            particles = [p.position for p in pso.swarm]
            particles_array = np.array(particles)
            
            ax.scatter(particles_array[:, 0], particles_array[:, 1], 
                      c='#2196F3', s=50, alpha=0.6, label='Particles')
            ax.scatter(best_position[0], best_position[1], 
                      c='#FF6B6B', s=200, marker='*', label='Best Position', zorder=5)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(f'PSO - Best Cost: {pso.history[-1] if pso.history else "N/A"}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax1 = self.fig.add_subplot(121)
            ax2 = self.fig.add_subplot(122)
            
            if hasattr(pso, 'history') and pso.history:
                ax1.plot(pso.history, 'b-', linewidth=2)
                ax1.set_xlabel('Iteration')
                ax1.set_ylabel('Best Cost')
                ax1.set_title(f'PSO Convergence (Dimensions: {dimensions})')
                ax1.grid(True, alpha=0.3)
            
            ax2.bar(range(len(best_position)), best_position, color='#FF9800')
            ax2.set_xlabel('Dimension')
            ax2.set_ylabel('Value')
            ax2.set_title('Best Position Values')
            ax2.grid(axis='y', alpha=0.3)
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def run_vrp(self):
        try:
            num_vehicles = self.get_param_value("Number of Vehicles:", "int")
            num_customers = self.get_param_value("Number of Customers:", "int")
            pop_size = self.get_param_value("Population Size:", "int")
            generations = self.get_param_value("Generations:", "int")
            mutation_rate = self.get_param_value("Mutation Rate:", "float")
            
            np.random.seed(42)
            depot = [0, 0]
            customers = {i: (np.random.rand()*10, np.random.rand()*10) for i in range(num_customers)}
            
            vrp = VRP(
                depot=depot,
                customers=customers,
                num_vehicles=num_vehicles,
                population_size=pop_size,
                generations=generations,
                mutation_rate=mutation_rate
            )
            
            self.results_text.insert(tk.END, "Running VRP GA...\n")
            self.root.update()
            
            best_solution = vrp.run()
            
            results = f"""
VRP Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━
Number of Vehicles: {num_vehicles}
Number of Customers: {num_customers}
Best Solution: {best_solution}
"""
            self.results_text.insert(tk.END, results)
            
            self.visualize_vrp(depot, customers, num_vehicles)
        
        except Exception as e:
            self.results_text.insert(tk.END, f"Error: {str(e)}\n")
    
    def visualize_vrp(self, depot, customers, num_vehicles):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        ax.scatter(depot[0], depot[1], s=400, c='#FF6B6B', marker='s', 
                  label='Depot', zorder=5, edgecolors='black', linewidth=2)
        
        colors = ['#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
        customer_positions = np.array(list(customers.values()))
        ax.scatter(customer_positions[:, 0], customer_positions[:, 1], 
                  s=150, c='#2196F3', label='Customers', alpha=0.7)
        
        for i, (pos_x, pos_y) in enumerate(customer_positions):
            ax.annotate(f'C{i}', (pos_x, pos_y), fontsize=9, ha='center')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Vehicle Routing Problem ({num_vehicles} Vehicles)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def run_nurse_scheduling(self):
        try:
            num_nurses = self.get_param_value("Number of Nurses:", "int")
            num_days = self.get_param_value("Number of Days:", "int")
            max_shifts = self.get_param_value("Max Shifts per Nurse:", "int")
            pop_size = self.get_param_value("Population Size:", "int")
            generations = self.get_param_value("Generations:", "int")
            mutation_rate = self.get_param_value("Mutation Rate:", "float")
            
            ns = NurseScheduling(
                num_nurses=num_nurses,
                days=num_days,
                shifts=[0, 1, 2, 3],
                max_shifts=max_shifts,
                population_size=pop_size,
                generations=generations,
                mutation_rate=mutation_rate
            )
            
            self.results_text.insert(tk.END, "Running Nurse Scheduling GA...\n")
            self.root.update()
            
            best_schedule = ns.run()
            
            results = f"""
Nurse Scheduling Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━
Number of Nurses: {num_nurses}
Number of Days: {num_days}
Max Shifts per Nurse: {max_shifts}
Schedule Found: ✓
Shifts per day: {np.sum(best_schedule, axis=0)}
"""
            self.results_text.insert(tk.END, results)
            
            self.visualize_nurse_scheduling(best_schedule, num_nurses, num_days)
        
        except Exception as e:
            self.results_text.insert(tk.END, f"Error: {str(e)}\n")
    
    def visualize_nurse_scheduling(self, schedule, num_nurses, num_days):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        im = ax.imshow(schedule, cmap='YlOrRd', aspect='auto')
        
        ax.set_xticks(range(num_days))
        ax.set_yticks(range(num_nurses))
        ax.set_xticklabels([f'Day {i}' for i in range(num_days)])
        ax.set_yticklabels([f'Nurse {i}' for i in range(num_nurses)])
        ax.set_xlabel('Days')
        ax.set_ylabel('Nurses')
        ax.set_title('Nurse Scheduling Solution')
        
        for i in range(num_nurses):
            for j in range(num_days):
                text = ax.text(j, i, int(schedule[i, j]),
                             ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, ax=ax, label='Shift Number')
        self.fig.tight_layout()
        self.canvas.draw()
    
    def run_feature_selection(self):
        try:
            num_samples = self.get_param_value("Number of Samples:", "int")
            num_features = self.get_param_value("Number of Features:", "int")
            pop_size = self.get_param_value("Population Size:", "int")
            generations = self.get_param_value("Generations:", "int")
            mutation_rate = self.get_param_value("Mutation Rate:", "float")
            
            X, y = make_friedman1(n_samples=num_samples, n_features=num_features, random_state=42)
            
            fs = FeatureSelection(
                X=X,
                y=y,
                model=LinearRegression(),
                task="regression",
                population_size=pop_size,
                generations=generations,
                mutation_rate=mutation_rate
            )
            
            self.results_text.insert(tk.END, "Running Feature Selection GA...\n")
            self.root.update()
            
            best_features = fs.run()
            
            selected_count = np.sum(best_features)
            results = f"""
Feature Selection Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Features: {num_features}
Selected Features: {selected_count}
Selection Ratio: {(selected_count/num_features)*100:.1f}%
Best Features: {np.where(best_features)[0].tolist()}
"""
            self.results_text.insert(tk.END, results)
            
            self.visualize_feature_selection(best_features, num_features)
        
        except Exception as e:
            self.results_text.insert(tk.END, f"Error: {str(e)}\n")
    
    def visualize_feature_selection(self, features, num_features):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        feature_names = [f'Feature {i}' for i in range(num_features)]
        colors = ['#4CAF50' if f == 1 else '#ccc' for f in features]
        
        ax.bar(feature_names, features, color=colors)
        ax.set_ylabel('Selected (1) / Not Selected (0)')
        ax.set_title('Feature Selection Results')
        ax.set_ylim(0, 1.2)
        
        if num_features > 10:
            ax.tick_params(axis='x', rotation=45)
        
        self.fig.tight_layout()
        self.canvas.draw()


def main():
    root = tk.Tk()
    app = AlgorithmGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
