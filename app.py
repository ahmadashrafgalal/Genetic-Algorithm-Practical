import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.datasets import make_friedman1
from sklearn.linear_model import LinearRegression

# Import your algorithms
from Algorithms.KnapSack import Knapsack
from Algorithms.PSO import PSO
from Algorithms.TSP import TSP 
from Algorithms.VRP import VRP 
from Algorithms.NQueen import NQueens
from Algorithms.NurseScheduling import NurseScheduling 
from Algorithms.GraphColoring import GraphColoring 
from Algorithms.FeatureSelection import FeatureSelection

# --- HACK TO SHOW MATPLOTLIB PLOTS IN STREAMLIT ---
# This overrides plt.show() inside your algorithm files so they render in the web UI
# instead of opening pop-up windows, without needing to change your original code!
original_show = plt.show
def st_show(*args, **kwargs):
    st.pyplot(plt.gcf())
    plt.clf()
plt.show = st_show
# --------------------------------------------------

# Configure the app
st.set_page_config(page_title="AI Optimization Visualizer", layout="wide", page_icon="🧬")

# Sidebar for navigation
st.sidebar.title("🧬 AI Algorithms")
st.sidebar.write("Select a problem to solve:")
algorithm = st.sidebar.radio(
    "Available Algorithms:",
    [
        "🎒 Knapsack Problem",
        "🗺️ Traveling Salesperson (TSP)",
        "🚚 Vehicle Routing (VRP)",
        "👑 N-Queens",
        "🏥 Nurse Scheduling",
        "🎨 Graph Coloring",
        "📊 Feature Selection",
        "🦅 Particle Swarm (PSO)"
    ]
)

st.title(algorithm)
# --- 1. Knapsack Problem ---
if algorithm == "🎒 Knapsack Problem":
    st.write("**What is this?** Imagine you have a backpack with a strict weight limit. You have various items, each with a different value and weight. The algorithm finds the best combination of items to pack to maximize your profit without breaking the bag!")
    
    # إضافة مربعات إدخال ليتمكن المستخدم من تغيير القيم بنفسه
    val_input = st.text_input("Enter the values of the items (separated by commas):", "50, 30, 20, 30, 50")
    wt_input = st.text_input("Enter the weights of the items (separated by commas):", "30, 50, 40, 20, 60")
    cap_input = st.number_input("Enter the maximum capacity of the backpack:", min_value=10, max_value=1000, value=110)

    if st.button("Run Algorithm"):
        with st.spinner("Finding the best items..."):
            items = ['A', 'B', 'C', 'D', 'E']
            
            # تحويل النص الذي أدخله المستخدم إلى قائمة من الأرقام
            try:
                values = [int(v.strip()) for v in val_input.split(',')]
                weights = [int(w.strip()) for w in wt_input.split(',')]
                
                # التأكد من أن عدد القيم يساوي عدد الأوزان ويساوي عدد العناصر
                if len(values) != len(items) or len(weights) != len(items):
                    st.error(f"Please ensure you enter exactly {len(items)} numbers for both values and weights.")
                else:
                    ga = Knapsack(items=items, values=values, weights=weights, capacity=cap_input)
                    best_solution = ga.run()
                    
                    st.success("Done!")
                    st.write(f"**Maximum Value Found:** {ga.fitness(best_solution)}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Fitness Improvement Over Time:**")
                        ga.plot_fitness()
                    with col2:
                        st.write("**Selected Items:**")
                        ga.plot_solution(best_solution)
            except ValueError:
                st.error("Please enter valid numbers separated by commas (e.g., 10, 20, 30).")

# --- 2. Traveling Salesperson (TSP) ---
elif algorithm == "🗺️ Traveling Salesperson (TSP)":
    st.write("**What is this?** Imagine a delivery person who needs to visit multiple cities and return home. The algorithm finds the absolute shortest route to save time and fuel.")
    
    # مربع نصي ليقوم المستخدم بإدخال الإحداثيات
    coords_input = st.text_input(
        "Enter city coordinates (x,y) separated by semicolons (;):", 
        "0,0; 1,3; 4,2; 3,0; 1,4"
    )
    
    if st.button("Run Algorithm"):
        with st.spinner("Calculating the shortest route..."):
            try:
                # تحويل النص إلى قائمة من الإحداثيات
                # مثال: "0,0; 1,3" يتحول إلى [[0.0, 0.0], [1.0, 3.0]]
                raw_points = coords_input.split(';')
                cities_coords = []
                for point in raw_points:
                    if point.strip(): # لتجاهل الفراغات
                        x, y = point.split(',')
                        cities_coords.append([float(x.strip()), float(y.strip())])
                
                # إنشاء أسماء المدن تلقائياً (A, B, C, D...) على حسب عدد الإحداثيات
                city_labels = [chr(65 + i) if i < 26 else f"C{i+1}" for i in range(len(cities_coords))]
                
                if len(cities_coords) < 3:
                    st.error("Please enter at least 3 cities to find a route.")
                else:
                    tsp = TSP(
                        cities_coords=cities_coords,
                        city_labels=city_labels
                    )
                    best = tsp.run()
                    result = tsp.get_result()
                    
                    st.success("Done!")
                    st.write(f"**Shortest Route Found:** {' ➡️ '.join(result['tour'])}")
                    st.write(f"**Total Distance:** {result['cost']:.2f}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Route Map:**")
                        tsp.plot_tour(best)
                    with col2:
                        st.write("**Learning Curve:**")
                        tsp.plot_convergence()
                        
            except ValueError:
                st.error("Invalid format! Please ensure you enter coordinates like this: 0,0; 1,3; 4,2")

# --- 3. Vehicle Routing (VRP) ---
elif algorithm == "🚚 Vehicle Routing (VRP)":
    st.write("**What is this?** Similar to the Traveling Salesperson, but here we have multiple delivery trucks (vehicles) starting from a main depot. The algorithm organizes routes for all trucks to serve all customers efficiently.")
    
    if st.button("Run Algorithm"):
        with st.spinner("Routing vehicles..."):
            depot = (0, 0)
            customers = {1: (2, 3), 2: (5, 4), 3: (1, 7), 4: (6, 8), 5: (3, 6)}
            vrp = VRP(depot=depot, customers=customers, num_vehicles=2)
            best = vrp.run()
            routes = vrp.get_routes(best)
            
            st.success("Done!")
            for i, r in enumerate(routes):
                st.write(f"🚐 **Vehicle {i+1} Route:** {' ➡️ '.join(['Depot'] + [f'Customer {x}' for x in r] + ['Depot'])}")
            
            col1, col2 = st.columns(2)
            with col1:
                vrp.plot(routes)
            with col2:
                vrp.plot_convergence()

# --- 4. N-Queens ---
elif algorithm == "👑 N-Queens":
    st.write("**What is this?** A classic puzzle. The goal is to place 8 Chess Queens on a chessboard so that none of them can attack each other. This means no two queens can share the same row, column, or diagonal.")
    
    if st.button("Run Algorithm"):
        with st.spinner("Placing queens..."):
            queens = NQueens(n=8, population_size=100, generations=200)
            best_solution = queens.run()
            
            st.success("Done!")
            conflicts = -queens.fitness(best_solution)
            st.write(f"**Remaining Conflicts (Threats):** {conflicts} (0 means perfect solution!)")
            
            col1, col2 = st.columns(2)
            with col1:
                queens.plot_board(best_solution)
            with col2:
                queens.plot_history()

# --- 5. Nurse Scheduling ---
elif algorithm == "🏥 Nurse Scheduling":
    st.write("**What is this?** Hospitals need to assign shifts to nurses. The algorithm organizes a schedule that covers all necessary hospital shifts while ensuring nurses aren't overworked.")
    
    if st.button("Run Algorithm"):
        with st.spinner("Generating schedule..."):
            scheduler = NurseScheduling(num_nurses=4, days=5)
            best_schedule = scheduler.run()
            
            st.success("Done!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Shift Heatmap:**")
                scheduler.plot_schedule(best_schedule)
            with col2:
                st.write("**Learning Curve:**")
                scheduler.plot_history()

# --- 6. Graph Coloring ---
elif algorithm == "🎨 Graph Coloring":
    st.write("**What is this?** Imagine a map or a network of connected dots. The algorithm tries to color every dot using a limited number of colors, ensuring that no two connected dots share the same color.")
    
    if st.button("Run Algorithm"):
        with st.spinner("Coloring the graph..."):
            G = nx.Graph()
            edges = [(0,1),(0,2),(1,2),(1,3),(2,3),(3,4),(4,5),(5,0)]
            G.add_edges_from(edges)
            
            gc = GraphColoring(graph=G, num_colors=3)
            best_solution = gc.run()
            
            st.success("Done!")
            st.write(f"**Colors assigned to nodes (0 to 5):** {best_solution}")
            
            col1, col2 = st.columns(2)
            with col1:
                gc.visualize(best_solution)
            with col2:
                gc.plot_history()

# --- 7. Feature Selection ---
elif algorithm == "📊 Feature Selection":
    st.write("**What is this?** In Artificial Intelligence, we often have too much data. This algorithm acts like a smart filter, picking only the most important features (variables) to make accurate predictions, ignoring the useless noise.")
    
    if st.button("Run Algorithm"):
        with st.spinner("Finding best features..."):
            X, y = make_friedman1(n_samples=200, n_features=10, noise=0.1)
            ga = FeatureSelection(X=X, y=y, model=LinearRegression(), task="regression")
            best = ga.run()
            
            st.success("Done!")
            features = ga.get_selected_features(best)
            st.write(f"**Selected Important Features:** {features}")
            
            col1, col2 = st.columns(2)
            with col1:
                ga.plot_features(best)
            with col2:
                ga.plot_history()

# --- 8. Particle Swarm (PSO) ---
elif algorithm == "🦅 Particle Swarm (PSO)":
    st.write("**What is this?** Inspired by a flock of birds finding food, this algorithm creates a 'swarm' of particles that fly around a complex mathematical landscape. They communicate with each other to find the absolute lowest point (the best solution).")
    
    if st.button("Run Algorithm"):
        with st.spinner("Swarm is searching..."):
            pso = PSO()
            best_position, best_fitness = pso.run()
            
            st.success("Done!")
            st.write(f"**Best Position Found:** {best_position}")
            st.write(f"**Lowest Value (Fitness):** {best_fitness:.6f}")
            
            pso.plot_history()