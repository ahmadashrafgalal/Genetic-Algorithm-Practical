import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.datasets import make_friedman1
from sklearn.linear_model import LinearRegression
import textwrap
import folium
from streamlit_folium import st_folium
from folium import plugins

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
# ضيف السطر ده في أول الملف خالص مع باقي الـ imports
import textwrap

# --- 1. Knapsack Problem ---
if algorithm == "🎒 Knapsack Problem":
    st.write("**What is this?** Imagine you have a backpack with a strict weight limit. You have various items, each with a different value and weight. The algorithm finds the best combination of items to pack to maximize your profit without breaking the bag!")
    
    val_input = st.text_input("Enter the values of the items (separated by commas):", "50, 30, 20, 30, 50")
    wt_input = st.text_input("Enter the weights of the items (separated by commas):", "30, 50, 40, 20, 60")
    cap_input = st.number_input("Enter the maximum capacity of the backpack:", min_value=10, max_value=1000, value=110)

    if st.button("Run Algorithm"):
        with st.spinner("Finding the best items..."):
            items = ['A', 'B', 'C', 'D', 'E']
            
            try:
                values = [int(v.strip()) for v in val_input.split(',')]
                weights = [int(w.strip()) for w in wt_input.split(',')]
                
                if len(values) != len(items) or len(weights) != len(items):
                    st.error(f"Please ensure you enter exactly {len(items)} numbers for both values and weights.")
                else:
                    # تشغيل الخوارزمية من ملف KnapSack.py
                    ga = Knapsack(items=items, values=values, weights=weights, capacity=cap_input)
                    best_solution = ga.run()
                    
                    st.success("Done!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Fitness Improvement Over Time:**")
                        ga.plot_fitness()
                    
                    with col2:
                        st.write("**Visualized Backpack:**")
                        
                        total_w = sum(weights[i] * best_solution[i] for i in range(len(items)))
                        total_v = sum(values[i] * best_solution[i] for i in range(len(items)))

                        rows_html = ""
                        for i in range(len(items)):
                            selected = best_solution[i]
                            w = weights[i] * selected
                            v = values[i] * selected
                            
                            # --- التعديل هنا: الألوان والشفافية ---
                            opacity = "1" if selected else "0.7"
                            bg_color = "#eafaf1" if selected else "#fdedec" # أخضر فاتح / أحمر فاتح
                            border_left = "5px solid #2ecc71" if selected else "5px solid #e74c3c" # حافة يسرى خضراء / حمراء
                            
                            # تمت إضافة border-left وإعطاء مسافة سفلية (margin-bottom) وحواف دائرية خفيفة
                            rows_html += f"""
<div style="display: flex; border-left: {border_left}; border-bottom: 1px solid #eee; margin-bottom: 3px; border-radius: 4px; padding: 12px 15px; opacity: {opacity}; background-color: {bg_color};">
<div style="flex: 1.2; font-weight: bold; color: #2c3e50; display: flex; align-items: center;">Item {items[i]}: <span style="font-size: 18px; margin-left: 5px;">{selected}</span></div>
<div style="flex: 1; text-align: center; border-left: 1px solid #ccc; padding-left: 10px;">
<div style="font-size: 10px; color: #7f8c8d; text-transform: uppercase;">Total Weight</div>
<div style="font-weight: bold; color: #34495e;">{weights[i]} <span style="font-size: 12px; font-weight: normal;">kg</span></div>
</div>
<div style="flex: 1; text-align: center; border-left: 1px solid #ccc; padding-left: 10px;">
<div style="font-size: 10px; color: #7f8c8d; text-transform: uppercase;">Total Value</div>
<div style="font-weight: bold; color: #34495e;">{values[i]}</div>
</div>
</div>
"""

                        # إزالة المسافات البادئة تماماً من هنا أيضاً
                        full_html = f"""
<div style="display: flex; justify-content: center; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
<div style="background-color: #eef2f5; border-radius: 30px; padding: 20px; width: 100%; max-width: 380px; border: 4px solid #dce4ec; box-shadow: 0 10px 20px rgba(0,0,0,0.1); position: relative;">
<div style="position: absolute; top: -15px; left: 50%; transform: translateX(-50%); width: 60px; height: 20px; border: 4px solid #dce4ec; border-bottom: none; border-radius: 20px 20px 0 0; background-color: #eef2f5;"></div>
<h3 style="text-align: center; color: #34495e; margin-top: 10px; margin-bottom: 15px; font-size: 16px; font-weight: 800; letter-spacing: 1px;">🎒 OPTIMAL ITEM MIX</h3>
<div style="background-color: white; border-radius: 15px; overflow: hidden; border: 2px solid #e1e4e8;">
{rows_html}
<div style="background-color: #c05621; color: white; display: flex; padding: 15px; align-items: center;">
<div style="flex: 1.2; font-size: 20px; font-weight: 900; letter-spacing: 1px;">TOTAL</div>
<div style="flex: 1; text-align: center; border-left: 1px solid rgba(255,255,255,0.3);">
<div style="font-size: 10px; text-transform: uppercase; opacity: 0.8;">Weight</div>
<div style="font-weight: bold; font-size: 18px;">{total_w} <span style="font-size: 12px; font-weight: normal;">kg</span></div>
</div>
<div style="flex: 1; text-align: center; border-left: 1px solid rgba(255,255,255,0.3);">
<div style="font-size: 10px; text-transform: uppercase; opacity: 0.8;">Value</div>
<div style="font-weight: bold; font-size: 18px;">{total_v}</div>
</div>
</div>
</div>
</div>
</div>
"""
                        
                        st.markdown(full_html, unsafe_allow_html=True)    
            except ValueError:
                st.error("Please enter valid numbers separated by commas (e.g., 10, 20, 30).")

# --- 2. Traveling Salesperson (TSP) ---
elif algorithm == "🗺️ Traveling Salesperson (TSP)":
    st.write("**What is this?** Imagine a delivery person who needs to visit multiple locations. Instead of typing coordinates, **click anywhere on the map** below to drop pins. The algorithm will find the absolute shortest route to connect all your pins!")
    
    # 1. تهيئة الذاكرة لحفظ النقاط وحالة عرض النتائج
    if "cities" not in st.session_state:
        st.session_state.cities = []
    if "show_tsp_results" not in st.session_state:
        st.session_state.show_tsp_results = False
    if "tsp_data" not in st.session_state:
        st.session_state.tsp_data = {}

    st.markdown("### 📍 Drop your delivery points here:")
    
    m_input = folium.Map(location=[29.0661, 31.0994], zoom_start=13)
    
    # رسم الدبابيس الموجودة بالفعل مع تمييز نقطة البداية
    for i, city in enumerate(st.session_state.cities):
        is_start = (i == 0)
        marker_color = "green" if is_start else "red"
        marker_icon = "play" if is_start else "info-sign"
        popup_text = "📍 Start Point (A)" if is_start else f"Point {chr(65+i)}"
        
        folium.Marker(
            location=city,
            popup=popup_text,
            icon=folium.Icon(color=marker_color, icon=marker_icon)
        ).add_to(m_input)

    map_data = st_folium(m_input, width=800, height=400, key="input_map")
    
    # عند إضافة مدينة جديدة، نلغي عرض النتائج القديمة
    if map_data and map_data.get("last_clicked"):
        lat = map_data["last_clicked"]["lat"]
        lng = map_data["last_clicked"]["lng"]
        if [lat, lng] not in st.session_state.cities:
            st.session_state.cities.append([lat, lng])
            st.session_state.show_tsp_results = False # إخفاء النتائج لطلب حساب جديد
            st.rerun()

    if st.button("🗑️ Clear Map") and len(st.session_state.cities) > 0:
        st.session_state.cities = []
        st.session_state.show_tsp_results = False
        st.rerun()

    # 3. تشغيل الخوارزمية وحفظ النتائج في الذاكرة
    if st.button("🚀 Run Algorithm"):
        if len(st.session_state.cities) < 3:
            st.error("Please click on the map to add at least 3 points first!")
        else:
            with st.spinner("AI is calculating the optimal route..."):
                cities_coords = st.session_state.cities
                city_labels = [chr(65+i) for i in range(len(cities_coords))]
                
                tsp = TSP(cities_coords=cities_coords, city_labels=city_labels)
                best_chrom = tsp.run()
                result = tsp.get_result()
                
                original_tour = list(range(len(cities_coords)))
                original_cost = tsp.cost(original_tour)
                
                optimized_tour = best_chrom
                optimized_cost = result['cost']
                
                saved_percent = 0
                if original_cost > 0:
                    saved_percent = ((original_cost - optimized_cost) / original_cost) * 100
                
                # حفظ كل النتائج في الذاكرة (Session State)
                st.session_state.tsp_data = {
                    "cities_coords": cities_coords.copy(),
                    "city_labels": city_labels.copy(),
                    "original_tour": original_tour,
                    "optimized_tour": optimized_tour,
                    "original_cost": original_cost,
                    "optimized_cost": optimized_cost,
                    "saved_percent": saved_percent
                }
                st.session_state.show_tsp_results = True

    # 4. عرض النتائج الثابتة بناءً على الذاكرة وليس على ضغطة الزر
    if st.session_state.show_tsp_results:
        data = st.session_state.tsp_data
        
        st.success("Optimization Complete!")
        
        stats_html = textwrap.dedent(f"""
        <div style="display: flex; justify-content: space-between; background-color: white; border-radius: 15px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px; border-left: 8px solid #3498db;">
            <div style="text-align: center;">
                <div style="color: #7f8c8d; font-size: 12px; font-weight: bold; text-transform: uppercase;">Original Distance</div>
                <div style="color: #e74c3c; font-size: 24px; font-weight: 900;">{data['original_cost']:.2f} <span style="font-size: 14px;">km</span></div>
            </div>
            <div style="text-align: center;">
                <div style="color: #7f8c8d; font-size: 12px; font-weight: bold; text-transform: uppercase;">Optimized Distance</div>
                <div style="color: #2ecc71; font-size: 24px; font-weight: 900;">{data['optimized_cost']:.2f} <span style="font-size: 14px;">km</span></div>
            </div>
            <div style="text-align: center;">
                <div style="color: #7f8c8d; font-size: 12px; font-weight: bold; text-transform: uppercase;">Distance Saved</div>
                <div style="color: #f39c12; font-size: 24px; font-weight: 900;">{data['saved_percent']:.1f}%</div>
            </div>
        </div>
        """)
        st.markdown(stats_html, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h4 style='text-align: center; color: #e74c3c;'>❌ Before (Your Order)</h4>", unsafe_allow_html=True)
            m_before = folium.Map(location=[29.0661, 31.0994], zoom_start=13)
            
            for i, city in enumerate(data['cities_coords']):
                is_start = (i == 0)
                marker_color = "green" if is_start else "red"
                marker_icon = "play" if is_start else "info-sign"
                popup_text = "📍 Start Point (A)" if is_start else f"Point {data['city_labels'][i]}"
                
                folium.Marker(
                    location=city,
                    popup=popup_text,
                    icon=folium.Icon(color=marker_color, icon=marker_icon)
                ).add_to(m_before)
            
            original_path = [(float(data['cities_coords'][i][0]), float(data['cities_coords'][i][1])) for i in data['original_tour']]
            original_path.append(original_path[0])  # Close the loop
            
            plugins.AntPath(locations=original_path, color="red", weight=5, delay=800).add_to(m_before)
            st_folium(m_before, width=400, height=350, key="map_before_result")

        with col2:
            st.markdown("<h4 style='text-align: center; color: #2ecc71;'>✨ After (Genetic Optimized)</h4>", unsafe_allow_html=True)
            m_after = folium.Map(location=[29.0661, 31.0994], zoom_start=13)
            
            for i, city in enumerate(data['cities_coords']):
                is_start = (i == 0)
                marker_color = "green" if is_start else "red"
                marker_icon = "play" if is_start else "info-sign"
                popup_text = "📍 Start Point (A)" if is_start else f"Point {data['city_labels'][i]}"
                
                folium.Marker(
                    location=city,
                    popup=popup_text,
                    icon=folium.Icon(color=marker_color, icon=marker_icon)
                ).add_to(m_after)
            
            optimized_path = [(float(data['cities_coords'][i][0]), float(data['cities_coords'][i][1])) for i in data['optimized_tour']]
            optimized_path.append(optimized_path[0])  # Close the loop
            
            plugins.AntPath(locations=optimized_path, color="green", weight=5, delay=400).add_to(m_after)
            st_folium(m_after, width=400, height=350, key="map_after_result")

# --- 3. Vehicle Routing (VRP) ---
elif algorithm == "🚚 Vehicle Routing (VRP)":
    st.write("**What is this?** Similar to the Traveling Salesperson, but here we have multiple delivery trucks starting from a main depot. **Click on the map** below. Your **FIRST click** will be the Main Depot (HQ), and the rest will be delivery locations. The AI will dispatch the vehicles efficiently!")
    
    # 1. تهيئة الذاكرة للـ VRP
    if "vrp_cities" not in st.session_state:
        st.session_state.vrp_cities = []
    if "show_vrp_results" not in st.session_state:
        st.session_state.show_vrp_results = False
    if "vrp_data" not in st.session_state:
        st.session_state.vrp_data = {}

    st.markdown("### 📍 Drop your Depot & Delivery Points:")
    
    # تحديد عدد العربيات
    num_vehicles = st.number_input("🚐 Number of Delivery Vehicles:", min_value=1, max_value=10, value=2)
    
    m_input = folium.Map(location=[29.0661, 31.0994], zoom_start=13)
    
    # رسم الدبابيس (النقطة الأولى Depot والباقي Customers)
    for i, city in enumerate(st.session_state.vrp_cities):
        is_depot = (i == 0)
        marker_color = "darkblue" if is_depot else "red"
        marker_icon = "home" if is_depot else "info-sign"
        popup_text = "🏢 Main Depot" if is_depot else f"Customer {i}"
        
        folium.Marker(
            location=city,
            popup=popup_text,
            icon=folium.Icon(color=marker_color, icon=marker_icon)
        ).add_to(m_input)

    map_data = st_folium(m_input, width=800, height=400, key="vrp_input_map")
    
    # حفظ الإحداثيات
    if map_data and map_data.get("last_clicked"):
        lat = map_data["last_clicked"]["lat"]
        lng = map_data["last_clicked"]["lng"]
        if [lat, lng] not in st.session_state.vrp_cities:
            st.session_state.vrp_cities.append([lat, lng])
            st.session_state.show_vrp_results = False
            st.rerun()

    if st.button("🗑️ Clear Map") and len(st.session_state.vrp_cities) > 0:
        st.session_state.vrp_cities = []
        st.session_state.show_vrp_results = False
        st.rerun()

    # 3. تشغيل الخوارزمية
    if st.button("🚀 Dispatch Vehicles (Run)"):
        if len(st.session_state.vrp_cities) < 3:
            st.error("Please add at least 1 Depot and 2 Customers (total 3 points).")
        else:
            with st.spinner("AI is calculating optimal routes for your fleet..."):
                # استخراج الـ Depot والـ Customers من الإدخالات
                raw_coords = st.session_state.vrp_cities
                depot = tuple(raw_coords[0])
                customers = {i+1: tuple(raw_coords[i+1]) for i in range(len(raw_coords)-1)}
                
                # تشغيل خوارزمية VRP
                vrp = VRP(depot=depot, customers=customers, num_vehicles=num_vehicles, population_size=40, generations=100)
                
                # المسار العشوائي (بترتيب الإدخال)
                naive_chrom = list(customers.keys())
                naive_cost = 1 / vrp.fitness(naive_chrom)
                naive_routes = vrp.get_routes(naive_chrom)
                
                # المسار المحسن
                best_chrom = vrp.run()
                optimized_cost = 1 / vrp.fitness(best_chrom)
                optimized_routes = vrp.get_routes(best_chrom)
                
                saved_percent = 0
                if naive_cost > 0:
                    saved_percent = ((naive_cost - optimized_cost) / naive_cost) * 100
                
                st.session_state.vrp_data = {
                    "depot": depot,
                    "customers": customers,
                    "naive_routes": naive_routes,
                    "optimized_routes": optimized_routes,
                    "naive_cost": naive_cost,
                    "optimized_cost": optimized_cost,
                    "saved_percent": saved_percent,
                    "num_vehicles": num_vehicles
                }
                st.session_state.show_vrp_results = True

    # 4. عرض النتائج الثابتة
    if st.session_state.show_vrp_results:
        data = st.session_state.vrp_data
        
        st.success(f"Fleet Dispatched! {data['num_vehicles']} vehicles routed.")
        
        stats_html = textwrap.dedent(f"""
        <div style="display: flex; justify-content: space-between; background-color: white; border-radius: 15px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px; border-left: 8px solid #9b59b6;">
            <div style="text-align: center;">
                <div style="color: #7f8c8d; font-size: 12px; font-weight: bold; text-transform: uppercase;">Original Fleet Distance</div>
                <div style="color: #e74c3c; font-size: 24px; font-weight: 900;">{data['naive_cost']:.2f} <span style="font-size: 14px;">km</span></div>
            </div>
            <div style="text-align: center;">
                <div style="color: #7f8c8d; font-size: 12px; font-weight: bold; text-transform: uppercase;">Optimized Fleet Distance</div>
                <div style="color: #2ecc71; font-size: 24px; font-weight: 900;">{data['optimized_cost']:.2f} <span style="font-size: 14px;">km</span></div>
            </div>
            <div style="text-align: center;">
                <div style="color: #7f8c8d; font-size: 12px; font-weight: bold; text-transform: uppercase;">Fuel/Distance Saved</div>
                <div style="color: #f39c12; font-size: 24px; font-weight: 900;">{data['saved_percent']:.1f}%</div>
            </div>
        </div>
        """)
        st.markdown(stats_html, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        # قائمة ألوان لتمييز كل سيارة
        route_colors = ['green', 'purple', 'orange', 'cadetblue', 'darkred', 'black', 'darkgreen', 'blue']
        
        with col1:
            st.markdown("<h4 style='text-align: center; color: #e74c3c;'>❌ Before (Naive Dispatch)</h4>", unsafe_allow_html=True)
            m_before = folium.Map(location=[29.0661, 31.0994], zoom_start=13)
            
            folium.Marker(data['depot'], popup="🏢 Depot", icon=folium.Icon(color="darkblue", icon="home")).add_to(m_before)
            for cid, coords in data['customers'].items():
                folium.Marker(coords, popup=f"Cust {cid}", icon=folium.Icon(color="red", icon="info-sign")).add_to(m_before)
            
            # رسم مسار كل سيارة قبل التحسين
            for idx, route in enumerate(data['naive_routes']):
                if not route: continue
                path = [data['depot']] + [data['customers'][c] for c in route] + [data['depot']]
                color = route_colors[idx % len(route_colors)]
                plugins.AntPath(locations=path, color=color, weight=5, delay=800, dash_array="10").add_to(m_before)
                
            st_folium(m_before, width=400, height=350, key="vrp_before")

        with col2:
            st.markdown("<h4 style='text-align: center; color: #2ecc71;'>✨ After (AI Optimized Dispatch)</h4>", unsafe_allow_html=True)
            m_after = folium.Map(location=[29.0661, 31.0994], zoom_start=13)
            
            folium.Marker(data['depot'], popup="🏢 Depot", icon=folium.Icon(color="darkblue", icon="home")).add_to(m_after)
            for cid, coords in data['customers'].items():
                folium.Marker(coords, popup=f"Cust {cid}", icon=folium.Icon(color="red", icon="info-sign")).add_to(m_after)
            
            # رسم مسار كل سيارة بعد التحسين بألوان مميزة
            for idx, route in enumerate(data['optimized_routes']):
                if not route: continue
                path = [data['depot']] + [data['customers'][c] for c in route] + [data['depot']]
                color = route_colors[idx % len(route_colors)]
                plugins.AntPath(locations=path, color=color, weight=5, delay=400).add_to(m_after)
                
            st_folium(m_after, width=400, height=350, key="vrp_after")

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