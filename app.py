import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.datasets import make_friedman1
from sklearn.linear_model import LinearRegression
import textwrap
import folium
from streamlit_folium import st_folium
from folium import plugins
import streamlit.components.v1 as components
import random

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

    # 1. تهيئة الذاكرة للرقعة العشوائية والحل
    if "nq_before" not in st.session_state:
        st.session_state.nq_before = [random.randint(0, 7) for _ in range(8)]
    if "nq_after" not in st.session_state:
        st.session_state.nq_after = None

    # دالة لبناء رقعة الشطرنج التفاعلية بالـ HTML/JS
    def get_chess_html(board, title):
        html = f"""
        <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; text-align: center;">
            <h3 style="color: #2c3e50; margin-bottom: 10px;">{title}</h3>
            <div style="display: inline-grid; grid-template-columns: repeat(8, 45px); border: 4px solid #2c3e50; box-shadow: 0 6px 12px rgba(0,0,0,0.3);">
        """
        for r in range(8):
            for c in range(8):
                # تحديد لون المربع (أبيض أو أسود)
                color_class = "white" if (r + c) % 2 == 0 else "black"
                bg_color = "#f0d9b5" if color_class == "white" else "#b58863"
                has_queen = (board[r] == c)
                
                # استخدام رمز الملكة الكلاسيكي
                content = "♛" if has_queen else ""
                
                # تصميم المربع وإضافة box-sizing لمنع تداخل الأبعاد
                html += f"""
                <div class="square" data-row="{r}" data-col="{c}"
                     style="width: 45px; height: 45px; background-color: {bg_color};
                            display: flex; align-items: center; justify-content: center;
                            font-size: 34px; color: #111; text-shadow: 0px 2px 4px rgba(255,255,255,0.5);
                            cursor: pointer; user-select: none; transition: all 0.15s ease-in-out; box-sizing: border-box;">
                     {content}
                </div>"""
                
        html += """
            </div>
            <p style="font-size: 13px; color: #7f8c8d; margin-top: 15px; font-weight: bold;">
                🖱️ Hover over any ♛ to see its attack paths!
            </p>
        </div>

        <script>
            const squares = document.querySelectorAll('.square');
            
            squares.forEach(sq => {
                sq.addEventListener('mouseenter', function() {
                    // لو المربع اللي وقفنا عليه جواه Queen
                    if (this.innerText.includes('♛')) {
                        const qRow = parseInt(this.dataset.row);
                        const qCol = parseInt(this.dataset.col);

                        squares.forEach(s => {
                            const r = parseInt(s.dataset.row);
                            const c = parseInt(s.dataset.col);

                            // لو المربع ده في نفس الصف أو العمود أو القطر
                            if (r === qRow || c === qCol || Math.abs(r - qRow) === Math.abs(c - qCol)) {
                                
                                // تحديد الملكة اللي الماوس واقف عليها بلون ذهبي مع عازل أسود رفيع
                                if (r === qRow && c === qCol) {
                                    s.style.boxShadow = 'inset 0 0 0 1px #B9F8CF, inset 0 0 0 4px #f1c40f, inset 0 0 15px rgba(241, 196, 15, 0.5)';
                                    return;
                                }

                                // لو المسار ده فيه Queen تانية -> إطار أحمر داخلي وتوهج وعازل أسود (Conflict)
                                if (s.innerText.includes('♛')) {
                                    s.style.boxShadow = 'inset 0 0 0 1px #B9F8CF, inset 0 0 0 4px #e74c3c, inset 0 0 15px rgba(231, 76, 60, 0.6)';
                                } 
                                // لو المسار فاضي -> إطار أخضر داخلي وتوهج وعازل أسود (Safe)
                                else {
                                    s.style.boxShadow = 'inset 0 0 0 1px #B9F8CF, inset 0 0 0 4px #2ecc71, inset 0 0 15px rgba(46, 204, 113, 0.4)';
                                }
                            }
                        });
                    }
                });

                // لما الماوس يبعد، نمسح الإطارات الداخلية كلها
                sq.addEventListener('mouseleave', function() {
                    squares.forEach(s => {
                        s.style.boxShadow = 'none';
                    });
                });
            });
        </script>
        """
        return html

    # أزرار التحكم
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🎲 Shuffle Random Board"):
            st.session_state.nq_before = [random.randint(0, 7) for _ in range(8)]
            st.session_state.nq_after = None
            st.rerun()
            
    with col_btn2:
        if st.button("🚀 Run AI Algorithm"):
            with st.spinner("AI is placing the queens..."):
                # تشغيل الخوارزمية من ملف NQueen.py
                queens = NQueens(n=8, population_size=100, generations=200)
                best_solution = queens.run()
                st.session_state.nq_after = best_solution

    st.markdown("<hr>", unsafe_allow_html=True)
    
    # عرض الرقعتين
    col_board1, col_board2 = st.columns(2)
    
    with col_board1:
        components.html(get_chess_html(st.session_state.nq_before, "❌ Before (Random)"), height=500)
        
    with col_board2:
        if st.session_state.nq_after is not None:
            components.html(get_chess_html(st.session_state.nq_after, "✨ After (AI Solved)"), height=500)

# --- 5. Nurse Scheduling ---
elif algorithm == "🏥 Nurse Scheduling":
    import numpy as np
    
    st.write("**What is this?** Hospitals need to assign shifts to nurses. The AI ensures every single day has full coverage for Morning, Evening, and Night shifts while keeping the workload fair.")

    if "ns_before" not in st.session_state:
        st.session_state.ns_before = None
    if "ns_after" not in st.session_state:
        st.session_state.ns_after = None
    if "ns_stats" not in st.session_state:
        st.session_state.ns_stats = None

    col_in1, col_in2 = st.columns(2)
    num_nurses = col_in1.number_input("👩‍⚕️ Number of Nurses", min_value=3, max_value=20, value=4)
    num_days = col_in2.number_input("📅 Number of Days", min_value=3, max_value=14, value=5)
    max_shifts_allowed = 4 

    def get_roster_html(schedule, title, is_optimized):
        header_color = "#2ecc71" if is_optimized else "#e74c3c"
        
        # 1. حساب المشاكل لكل ممرضة ولكل يوم لربطها بالـ Hover
        uncovered_days = []
        for d in range(num_days):
            col_data = schedule[:, d]
            is_uncovered = not (1 in col_data and 2 in col_data and 3 in col_data)
            uncovered_days.append("true" if is_uncovered else "false")
            
        overworked_nurses = []
        for n in range(num_nurses):
            is_overworked = np.count_nonzero(schedule[n]) > max_shifts_allowed
            overworked_nurses.append("true" if is_overworked else "false")
        
        # 2. تعديل الـ CSS لإضافة الألوان الشرطية
        html = f"""
        <style>
            .roster-table th, .roster-table td {{ transition: background-color 0.2s; }}
            .highlight-row {{ background-color: #e8f4f8 !important; }}
            .highlight-col {{ background-color: #e8f4f8 !important; }}
            .header-hover {{ cursor: pointer; background-color: #34495e !important; }}
            
            /* ألوان الهوفر للممرضين (أخضر لو سليم، أحمر لو Overworked) */
            .nurse-header[data-overworked="false"]:hover {{ background-color: #1abc9c !important; }}
            .nurse-header[data-overworked="true"]:hover {{ background-color: #e74c3c !important; }}
            
            /* ألوان الهوفر للأيام (أخضر لو سليم، أحمر لو Uncovered) */
            .day-header[data-uncovered="false"]:hover {{ background-color: #1abc9c !important; }}
            .day-header[data-uncovered="true"]:hover {{ background-color: #e74c3c !important; }}
        </style>
        <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
            <h3 style="color: {header_color}; text-align: center; margin-bottom: 15px;">{title}</h3>
            <div style="border-radius: 10px; overflow: hidden; box-shadow: 0 4px 8px rgba(0,0,0,0.1); border: 1px solid #ddd;">
                <table class="roster-table" style="width: 100%; border-collapse: collapse; background-color: white;">
                    <thead>
                        <tr style="background-color: #2c3e50; color: white;">
                            <th style="padding: 12px; border: 1px solid #34495e;">Nurse \ Day</th>
        """
        
        # إضافة الـ data-uncovered لرؤوس الأيام
        for d in range(num_days):
            html += f'<th class="day-header header-hover" data-col="{d+1}" data-uncovered="{uncovered_days[d]}" style="padding: 12px; border: 1px solid #34495e;">Day {d+1}</th>'
        html += "</tr></thead><tbody>"

        badges = {
            0: '<span style="color: #a4b0be; font-weight: bold;">🛏️ OFF</span>',
            1: '<span style="background-color: #fff2cc; color: #f39c12; padding: 4px 8px; border-radius: 10px; font-size: 11px;">☀️ MORN</span>',
            2: '<span style="background-color: #ffeaa7; color: #d35400; padding: 4px 8px; border-radius: 10px; font-size: 11px;">🌆 EVE</span>',
            3: '<span style="background-color: #c8d6e5; color: #222f3e; padding: 4px 8px; border-radius: 10px; font-size: 11px;">🌙 NIGHT</span>'
        }

        # إضافة الـ data-overworked لأسماء الممرضين
        for n in range(num_nurses):
            html += f'<tr class="nurse-row"><td class="nurse-header header-hover" data-overworked="{overworked_nurses[n]}" style="padding: 10px; font-weight: bold; color: white; border: 1px solid #34495e; text-align: center; background-color: #34495e;">👩‍⚕️ Nurse {chr(65+n)}</td>'
            for d in range(num_days):
                shift_val = schedule[n][d]
                html += f'<td class="cell-data col-{d+1}" style="padding: 8px; text-align: center; border: 1px solid #eee;">{badges[shift_val]}</td>'
            html += "</tr>"

        html += """
                </tbody>
            </table>
            </div>
            <p style="text-align: center; font-size: 12px; color: #7f8c8d; margin-top: 10px; font-weight: bold;">
                💡 Tip: <span style="color:#1abc9c">Hover</span> over Headers. <span style="color:#e74c3c">Red Hover</span> means a constraint is violated!
            </p>
        </div>

        <script>
            // تظليل الصف عند الوقوف على اسم الممرض (باقي الصف بياخد لون التظليل العادي)
            document.querySelectorAll('.nurse-header').forEach(header => {
                header.addEventListener('mouseenter', () => header.parentElement.classList.add('highlight-row'));
                header.addEventListener('mouseleave', () => header.parentElement.classList.remove('highlight-row'));
            });

            // تظليل العمود عند الوقوف على رأس اليوم (باقي العمود بياخد لون التظليل العادي)
            document.querySelectorAll('.day-header').forEach(header => {
                header.addEventListener('mouseenter', () => {
                    const col = header.dataset.col;
                    document.querySelectorAll('.col-' + col).forEach(cell => cell.classList.add('highlight-col'));
                });
                header.addEventListener('mouseleave', () => {
                    const col = header.dataset.col;
                    document.querySelectorAll('.col-' + col).forEach(cell => cell.classList.remove('highlight-col'));
                });
            });
        </script>
        """
        return html

    if st.button("🚀 Generate Smart Roster"):
        with st.spinner("AI is organizing the shifts..."):
            random_schedule = np.random.choice([0, 1, 2, 3], size=(num_nurses, num_days))
            
            scheduler = NurseScheduling(num_nurses=num_nurses, days=num_days, population_size=30, generations=100)
            best_schedule = scheduler.run()
            
            def calculate_stats(sched):
                overworked = 0
                uncovered_days = 0
                for nurse in sched:
                    if np.count_nonzero(nurse) > max_shifts_allowed:
                        overworked += 1
                for d in range(num_days):
                    column = sched[:, d]
                    if not (1 in column and 2 in column and 3 in column):
                        uncovered_days += 1
                return overworked, uncovered_days

            r_ov, r_un = calculate_stats(random_schedule)
            o_ov, o_un = calculate_stats(best_schedule)
            
            st.session_state.ns_before = random_schedule
            st.session_state.ns_after = best_schedule
            st.session_state.ns_stats = {"r_ov": r_ov, "r_un": r_un, "o_ov": o_ov, "o_un": o_un}

    if st.session_state.ns_after is not None:
        s = st.session_state.ns_stats
        
        stats_html = textwrap.dedent(f"""
        <div style="display: flex; justify-content: space-around; background-color: white; border-radius: 15px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 25px; border-left: 8px solid #8e44ad;">
            <div style="text-align: center; width: 45%;">
                <h4 style="color: #e74c3c;">❌ Random Schedule</h4>
                <p style="margin:0; font-size:14px;">Overworked: <b>{s['r_ov']}</b> | Uncovered Days: <b>{s['r_un']}</b></p>
            </div>
            <div style="text-align: center; width: 45%;">
                <h4 style="color: #2ecc71;">✨ AI Optimized</h4>
                <p style="margin:0; font-size:14px;">Overworked: <b>{s['o_ov']}</b> | Uncovered Days: <b>{s['o_un']}</b></p>
            </div>
        </div>
        """)
        st.markdown(stats_html, unsafe_allow_html=True)

        components.html(get_roster_html(st.session_state.ns_before, "❌ Before (Random)", False), height=400)
        st.markdown("<br>", unsafe_allow_html=True)
        components.html(get_roster_html(st.session_state.ns_after, "✨ After (AI Optimized)", True), height=400)

# --- 6. Graph Coloring ---
elif algorithm == "🎨 Graph Coloring":
    import numpy as np
    import math
    import random
    import networkx as nx
    
    st.write("**What is this?** The goal is to color the nodes of a graph so that **no two connected nodes share the same color**. The AI tries to resolve all conflicts while using the minimum number of colors.")

    # 1. تهيئة الذاكرة
    if "gc_matrix" not in st.session_state:
        st.session_state.gc_matrix = None
        st.session_state.gc_before = None
        st.session_state.gc_after = None
        st.session_state.gc_stats = None

    # 2. أدوات بناء الشبكة
    col_in1, col_in2 = st.columns(2)
    num_nodes = col_in1.slider("🔵 Number of Nodes", min_value=5, max_value=20, value=8)
    density = col_in2.slider("🕸️ Connection Density", min_value=0.2, max_value=1.0, value=0.4, step=0.1)

    # 3. دالة بناء الـ SVG التفاعلي (HTML/JS)
    def get_graph_html(adj_matrix, colors, title, is_optimized):
        header_color = "#2ecc71" if is_optimized else "#e74c3c"
        n = len(adj_matrix)
        
        palette = [
            "#3498db", "#e74c3c", "#f1c40f", "#2ecc71", "#9b59b6",
            "#e67e22", "#1abc9c", "#34495e", "#ff9ff3", "#feca57",
            "#ff6b6b", "#48dbfb", "#1dd1a1", "#5f27cd", "#c8d6e5",
            "#ff9f43", "#01a3a4", "#ff6b81", "#7bed9f", "#70a1ff"
        ]
        
        width = 400
        height = 400
        center_x, center_y = width // 2, height // 2
        radius = 150
        
        coords = []
        for i in range(n):
            angle = 2 * math.pi * i / n
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            coords.append((x, y))
            
        html = f"""
        <style>
            .graph-container {{ position: relative; width: {width}px; height: {height}px; margin: 0 auto; background-color: #fdfefe; border-radius: 15px; border: 1px solid #eee; box-shadow: 0 4px 10px rgba(0,0,0,0.05); }}
            .edge {{ transition: stroke 0.3s, opacity 0.3s, stroke-width 0.3s; stroke-linecap: round; }}
            
            .node-group {{ transition: transform 0.2s ease-out; cursor: pointer; }}
            .node-group:hover {{ transform: scale(1.2); }}
            
            @keyframes pulse-conflict {{
                0% {{ stroke-width: 2px; opacity: 0.8; stroke: #e74c3c; }}
                50% {{ stroke-width: 6px; opacity: 1; stroke: #ff0000; filter: drop-shadow(0 0 5px rgba(231,76,60,0.8)); }}
                100% {{ stroke-width: 2px; opacity: 0.8; stroke: #e74c3c; }}
            }}
            
            .conflict-edge {{ animation: pulse-conflict 1s infinite; }}
            
            /* لما نعمل هوفر والخط يطفي، نوقف النبض عشان التشتيت */
            .conflict-edge.dimmed {{ animation: none !important; opacity: 0.1 !important; stroke: #e74c3c !important; stroke-width: 2px !important; }}
            
            /* الخطوط العادية لما تتنور */
            .normal-edge.highlight-edge {{ stroke-width: 4px !important; opacity: 1 !important; stroke: #34495e !important; }}
            
            /* خطوط المشاكل لما تتنور (نحافظ على النبض والأحمر) */
            .conflict-edge.highlight-edge {{ opacity: 1 !important; }}
            
            .dimmed {{ opacity: 0.1 !important; }}
        </style>
        
        <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; text-align: center;">
            <h3 style="color: {header_color}; margin-bottom: 15px;">{title}</h3>
            <div class="graph-container">
                <svg width="{width}" height="{height}" style="overflow: visible;">
        """
        
        conflicts = 0
        edges_html = ""
        for i in range(n):
            for j in range(i+1, n):
                if adj_matrix[i][j] == 1:
                    x1, y1 = coords[i]
                    x2, y2 = coords[j]
                    
                    is_conflict = (colors[i] == colors[j])
                    if is_conflict: conflicts += 1
                    
                    edge_class = "edge conflict-edge" if is_conflict else "edge normal-edge"
                    edge_color = "#e74c3c" if is_conflict else "#bdc3c7"
                    edge_width = "3" if is_conflict else "1.5"
                    
                    edges_html += f'<line class="{edge_class}" data-source="{i}" data-target="{j}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{edge_color}" stroke-width="{edge_width}"></line>'
        
        # وضعنا الخطوط في طبقة (Group) لوحدها
        html += '<g class="edges-layer">' + edges_html + '</g>'
        
        nodes_html = ""
        for i in range(n):
            x, y = coords[i]
            color = palette[colors[i] % len(palette)]
            
            nodes_html += f"""
            <g class="node-group" data-id="{i}" style="transform-origin: {x}px {y}px;">
                <circle cx="{x}" cy="{y}" r="18" fill="{color}" stroke="#fff" stroke-width="3" filter="drop-shadow(0 2px 4px rgba(0,0,0,0.2))"></circle>
                <text x="{x}" y="{y+5}" font-size="12" font-weight="bold" fill="#fff" text-anchor="middle" pointer-events="none">{i}</text>
            </g>
            """
            
        # وضعنا العقد في طبقة (Group) لوحدها عشان تفضل دايماً فوق الخطوط
        html += '<g class="nodes-layer">' + nodes_html + '</g>'
        
        html += """
                </svg>
            </div>
            <p style="text-align: center; font-size: 13px; color: #7f8c8d; margin-top: 15px; font-weight: bold;">
                🖱️ Hover over any Node to isolate its connections. <span style="color:#e74c3c">Pulsing Red lines</span> are conflicts!
            </p>
        </div>

        <script>
            const nodeGroups = document.querySelectorAll('.node-group');
            const edges = document.querySelectorAll('.edge');

            nodeGroups.forEach(group => {
                group.addEventListener('mouseenter', function() {
                    const id = this.getAttribute('data-id');
                    
                    nodeGroups.forEach(g => g.classList.add('dimmed'));
                    edges.forEach(e => e.classList.add('dimmed'));
                    
                    this.classList.remove('dimmed');
                    
                    edges.forEach(e => {
                        const source = e.getAttribute('data-source');
                        const target = e.getAttribute('data-target');
                        
                        if (source === id || target === id) {
                            e.classList.remove('dimmed');
                            e.classList.add('highlight-edge');
                            
                            // السطر السحري: سحب الخط ليظهر فوق باقي الخطوط
                            e.parentNode.appendChild(e);
                            
                            const neighborId = source === id ? target : source;
                            document.querySelector(`.node-group[data-id="${neighborId}"]`).classList.remove('dimmed');
                        }
                    });
                });

                group.addEventListener('mouseleave', function() {
                    nodeGroups.forEach(g => g.classList.remove('dimmed'));
                    edges.forEach(e => {
                        e.classList.remove('dimmed');
                        e.classList.remove('highlight-edge');
                    });
                });
            });
        </script>
        """
        return html, conflicts

    # 4. أزرار التحكم
    btn_col1, btn_col2 = st.columns(2)
    
    with btn_col1:
        if st.button("🎲 Generate Random Graph"):
            adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
            for i in range(num_nodes):
                for j in range(i+1, num_nodes):
                    if random.random() < density:
                        adj_matrix[i][j] = 1
                        adj_matrix[j][i] = 1
                        
            random_colors = [random.randint(0, max(2, num_nodes//2)) for _ in range(num_nodes)]
            
            st.session_state.gc_matrix = adj_matrix
            st.session_state.gc_before = random_colors
            st.session_state.gc_after = None
            st.rerun()

    with btn_col2:
        if st.button("🚀 Optimize Colors (Run AI)"):
            if st.session_state.gc_matrix is None:
                st.error("Please Generate a Random Graph first!")
            else:
                with st.spinner("AI is resolving conflicts and minimizing colors..."):
                    matrix = st.session_state.gc_matrix
                    nx_graph = nx.from_numpy_array(matrix)
                    
                    gc_algo = GraphColoring(graph=nx_graph, population_size=50, generations=100)
                    best_coloring = gc_algo.run()
                    
                    st.session_state.gc_after = best_coloring
                    st.rerun()

    # 5. عرض النتائج
    if st.session_state.gc_matrix is not None:
        st.markdown("<hr>", unsafe_allow_html=True)
        
        matrix = st.session_state.gc_matrix
        before_colors = st.session_state.gc_before
        after_colors = st.session_state.gc_after
        
        total_edges = int(np.sum(matrix) / 2)
        
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            html_before, conflicts_before = get_graph_html(matrix, before_colors, "❌ Before (Random Colors)", False)
            used_colors_before = len(set(before_colors))
            
            st.markdown(f"""
            <div style="background-color: #fdedec; padding: 15px; border-radius: 10px; text-align: center; border-left: 5px solid #e74c3c; margin-bottom: 15px;">
                <div style="color: #c0392b; font-weight: bold;">🚨 Conflicts: <span style="font-size: 20px;">{conflicts_before}</span></div>
                <div style="color: #7f8c8d; font-size: 14px;">Colors Used: {used_colors_before} | Edges: {total_edges}</div>
            </div>
            """, unsafe_allow_html=True)
            
            components.html(html_before, height=550)

        with col_g2:
            if after_colors is not None:
                html_after, conflicts_after = get_graph_html(matrix, after_colors, "✨ After (AI Optimized)", True)
                used_colors_after = len(set(after_colors))
                
                st.markdown(f"""
                <div style="background-color: #eafaf1; padding: 15px; border-radius: 10px; text-align: center; border-left: 5px solid #2ecc71; margin-bottom: 15px;">
                    <div style="color: #27ae60; font-weight: bold;">✅ Conflicts: <span style="font-size: 20px;">{conflicts_after}</span></div>
                    <div style="color: #7f8c8d; font-size: 14px;">Colors Used: {used_colors_after} | Edges: {total_edges}</div>
                </div>
                """, unsafe_allow_html=True)
                
                components.html(html_after, height=550)

# --- 7. Feature Selection ---
elif algorithm == "📊 Feature Selection":
    import pandas as pd
    import time
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import LabelEncoder
    from Algorithms.FeatureSelection import FeatureSelection 
    import streamlit.components.v1 as components

    st.write("**What is this?** The AI acts as a smart filter for a **Bank Loan Approval System**. Instead of overwhelming the predictive model with all 13 customer details, the Genetic Algorithm finds the perfect subset of features to maximize accuracy and drop the noise.")

    # 1. تهيئة الذاكرة للنتائج الحقيقية
    if "fs_run" not in st.session_state:
        st.session_state.fs_run = False

    # قائمة الميزات بالأيقونات الحقيقية من FontAwesome
    features_list = [
        {"id": "credit.policy", "icon": "<i class='fas fa-file-contract'></i>"},
        {"id": "purpose", "icon": "<i class='fas fa-bullseye'></i>"},
        {"id": "int.rate", "icon": "<i class='fas fa-chart-line'></i>"},
        {"id": "installment", "icon": "<i class='fas fa-money-bill-wave'></i>"},
        {"id": "log.annual.inc", "icon": "<i class='fas fa-sack-dollar'></i>"},
        {"id": "dti", "icon": "<i class='fas fa-balance-scale'></i>"},
        {"id": "fico", "icon": "<i class='fas fa-star'></i>"},
        {"id": "days.with.cr.line", "icon": "<i class='fas fa-calendar-alt'></i>"},
        {"id": "revol.bal", "icon": "<i class='fas fa-credit-card'></i>"},
        {"id": "revol.util", "icon": "<i class='fas fa-chart-pie'></i>"},
        {"id": "inq.last.6mths", "icon": "<i class='fas fa-search'></i>"},
        {"id": "delinq.2yrs", "icon": "<i class='fas fa-exclamation-triangle'></i>"},
        {"id": "pub.rec", "icon": "<i class='fas fa-landmark'></i>"}
    ]

    # 2. دالة بناء البطاقات (مكتوبة بدون مسافات بادئة لمنع مشكلة الـ Code Box)
    def get_features_html(features, is_optimized, selected_subset=None):
        header_color = "#2ecc71" if is_optimized else "#e74c3c"
        title = "✨ AI Optimizing..." if is_optimized else "❌ Initial State (Using All Features)"
        
        if selected_subset is None:
            selected_subset = [f["id"] for f in features]

        html = "<link rel='stylesheet' href='https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css'>"
        html += "<style>"
        html += ".features-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(100px, 1fr)); gap: 10px; padding: 15px; background: #f8f9fa; border-radius: 10px; border: 1px solid #eee; }"
        html += ".f-card { background: white; border: 2px solid #bdc3c7; border-radius: 8px; padding: 10px; text-align: center; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; transition: all 0.3s ease; height: 75px; display: flex; flex-direction: column; justify-content: center; align-items: center; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }"
        html += ".f-icon { font-size: 24px; margin-bottom: 8px; }"
        html += ".f-name { font-size: 11px; font-weight: bold; color: #34495e; word-wrap: break-word; }"
        html += ".card-active { border-color: #3498db; background-color: #ebf5fb; color: #3498db; }"
        html += ".card-selected { border-color: #2ecc71; background-color: #eafaf1; border-width: 3px; color: #2ecc71; }"
        html += "@keyframes shake { 0% { transform: translateX(0); } 25% { transform: translateX(-3px); } 50% { transform: translateX(3px); } 75% { transform: translateX(-3px); } 100% { transform: translateX(0); } }"
        html += ".card-dropped { border-color: #bdc3c7; background-color: #fdfefe; opacity: 0.4; filter: grayscale(100%); color: #7f8c8d; }"
        html += ".card-dropped:hover { animation: shake 0.3s ease-in-out; opacity: 0.8; filter: grayscale(0%); border-color: #e74c3c; }"
        html += "</style>"

        html += "<div style='margin-bottom: 20px;'>"
        html += f"<h4 style='color: {header_color}; text-align: center; margin-bottom: 10px;'>{title}</h4>"
        html += "<div class='features-grid'>"
        
        for f in features:
            if not is_optimized:
                card_class = "card-active"
            else:
                is_selected = f['id'] in selected_subset
                card_class = "card-selected" if is_selected else "card-dropped"
                
            html += f"<div class='f-card {card_class}'><div class='f-icon'>{f['icon']}</div><div class='f-name'>{f['id']}</div></div>"
            
        html += "</div></div>"
        return html

    # دالة الإحصائيات (مكتوبة بشكل متصل لمنع الـ Code Box)
    def get_stats_html(gen_text, current_features, orig_acc_pct, current_acc, speed_gain):
        html = "<div style='display: flex; justify-content: space-around; background-color: white; border-radius: 15px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 25px; border-left: 8px solid #f39c12; font-family: sans-serif;'>"
        html += "<div style='text-align: center; border-right: 2px solid #eee; padding-right: 15px; width: 25%;'>"
        html += "<div style='color: #7f8c8d; font-size: 12px; font-weight: bold; text-transform: uppercase;'>Generation</div>"
        html += f"<div style='font-size: 24px; font-weight: 900; color: #f39c12;'>{gen_text}</div></div>"
        html += "<div style='text-align: center; border-right: 2px solid #eee; padding-right: 15px; width: 25%;'>"
        html += "<div style='color: #7f8c8d; font-size: 12px; font-weight: bold; text-transform: uppercase;'>Features Used</div>"
        html += f"<div style='font-size: 24px; font-weight: 900;'><span style='color: #e74c3c;'>13</span> ➔ <span style='color: #2ecc71;'>{current_features}</span></div></div>"
        html += "<div style='text-align: center; border-right: 2px solid #eee; padding-right: 15px; width: 25%;'>"
        html += "<div style='color: #7f8c8d; font-size: 12px; font-weight: bold; text-transform: uppercase;'>Model Accuracy (F1)</div>"
        html += f"<div style='font-size: 24px; font-weight: 900;'><span style='color: #e74c3c;'>{orig_acc_pct:.1f}%</span> ➔ <span style='color: #2ecc71;'>{current_acc:.1f}%</span></div></div>"
        html += "<div style='text-align: center; width: 25%;'>"
        html += "<div style='color: #7f8c8d; font-size: 12px; font-weight: bold; text-transform: uppercase;'>Est. Speed Gain</div>"
        html += f"<div style='color: #2ecc71; font-size: 24px; font-weight: 900;'>⚡ {speed_gain:.1f}%</div></div>"
        html += "</div>"
        return html

    # 3. زر التشغيل (التنفيذ الديناميكي مع تحديث الـ UI لحظياً)
    if st.button("🚀 Run Real Genetic Feature Selection"):
        try:
            # قراءة ومعالجة الداتا
            df = pd.read_csv("loan_data.csv")
            if 'purpose' in df.columns and df['purpose'].dtype == 'object':
                df['purpose'] = LabelEncoder().fit_transform(df['purpose'])
            X = df.drop('not.fully.paid', axis=1)
            y = df['not.fully.paid']

            # تهيئة الخوارزمية (5 أجيال)
            rf_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            fs_algo = FeatureSelection(X=X.values, y=y.values, model=rf_model, task="classification", population_size=10, generations=5)

            # حساب الدقة الأصلية
            rf_original = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            rf_original.fit(fs_algo.X_train, fs_algo.y_train)
            original_accuracy = accuracy_score(fs_algo.y_test, rf_original.predict(fs_algo.X_test))
            orig_acc_pct = original_accuracy * 100

            # أماكن الـ UI الديناميكية
            lottie_placeholder = st.empty() 
            stats_placeholder = st.empty()
            cards_placeholder = st.empty()

            # تشغيل Lottie Animation أثناء المعالجة
            lottie_html = """
            <script src="https://unpkg.com/@lottiefiles/dotlottie-wc@0.9.10/dist/dotlottie-wc.js" type="module"></script>
            <div style="display: flex; justify-content: center; align-items: center; height: 100%;">
                <dotlottie-wc src="https://lottie.host/662e5b74-efb0-49e7-8ddd-0c1cb263ef0b/XEWJlmt9b2.lottie" style="width: 250px; height: 250px" autoplay loop></dotlottie-wc>
            </div>
            """
            with lottie_placeholder.container():
                components.html(lottie_html, height=260)

            # 4. عرض النتائج المبنية على المعالجة الحقيقية (البداية)
            stats_placeholder.markdown(get_stats_html("Start", 13, orig_acc_pct, orig_acc_pct, 0.0), unsafe_allow_html=True)
            cards_placeholder.markdown(get_features_html(features_list, False), unsafe_allow_html=True)

            # اللوب بتاع الخوارزمية
            population = fs_algo.create_population()
            for gen in range(fs_algo.generations):
                new_population = []
                for _ in range(fs_algo.population_size):
                    p1 = fs_algo.selection(population)
                    p2 = fs_algo.selection(population)
                    child = fs_algo.crossover(p1, p2)
                    child = fs_algo.mutation(child)
                    new_population.append(child)
                population = new_population
                
                # استخراج أفضل حل في الجيل
                best_sol = max(population, key=fs_algo.fitness)
                current_acc = fs_algo.fitness(best_sol) * 100
                selected_idx = fs_algo.get_selected_features(best_sol)
                current_best_features = [X.columns[i] for i in selected_idx]
                
                # حماية برمجية
                if len(current_best_features) == 0:
                    current_best_features = list(X.columns)
                    
                speed_gain = ((13 - len(current_best_features)) / 13) * 100

                # تحديث الشاشة مع الجيل الجديد
                stats_placeholder.markdown(get_stats_html(f"#{gen+1}", len(current_best_features), orig_acc_pct, current_acc, speed_gain), unsafe_allow_html=True)
                cards_placeholder.markdown(get_features_html(features_list, True, current_best_features), unsafe_allow_html=True)
                
            # إخفاء الأنيميشن بعد انتهاء الـ 5 أجيال
            lottie_placeholder.empty()
            st.success("Evolution finished! Final optimized model is ready.")
            st.session_state.fs_run = True

        except Exception as e:
            st.error(f"Error: {e}")

# --- 8. Particle Swarm Optimization (PSO) ---
elif algorithm == "🦅 Particle Swarm (PSO)":
    import numpy as np
    class PSORecorder:

        def __init__(self, pso):
            self.pso = pso
            self.particles_history = []
            self.gbest_history = []

        def run(self):

            self.pso.initialize_swarm()

            swarm = self.pso.swarm

            gbest_particle = min(
                swarm,
                key=lambda p: p.best_fitness
            )

            gbest_position = gbest_particle.best_position.copy()

            gbest_fitness = gbest_particle.best_fitness

            for _ in range(self.pso.max_iter):

                for p in swarm:

                    r1 = np.random.rand(self.pso.dimensions)
                    r2 = np.random.rand(self.pso.dimensions)

                    p.velocity = (
                        self.pso.w * p.velocity
                        + self.pso.c1 * r1 * (p.best_position - p.position)
                        + self.pso.c2 * r2 * (gbest_position - p.position)
                    )

                    p.position += p.velocity
                    p.position = np.clip(
                        p.position,
                        self.pso.min_bound,
                        self.pso.max_bound
                    )

                    p.fitness = self.pso.cost_function(p.position)

                    if p.fitness < p.best_fitness:
                        p.best_position = p.position.copy()
                        p.best_fitness = p.fitness

                    if p.fitness < gbest_fitness:
                        gbest_fitness = p.fitness
                        gbest_position = p.position.copy()

                self.particles_history.append(
                    np.array([p.position.copy() for p in swarm])
                )

                self.gbest_history.append(
                    gbest_position.copy()
                )

    def build_3d_pso_animation(recorder, func, bounds):

        import plotly.graph_objects as go
        import numpy as np

        x = np.linspace(bounds[0], bounds[1], 80)
        y = np.linspace(bounds[0], bounds[1], 80)

        X, Y = np.meshgrid(x, y)
        Z = np.array([[func([xi, yi]) for xi, yi in zip(row_x, row_y)]
                    for row_x, row_y in zip(X, Y)])

        frames = []

        for i in range(len(recorder.particles_history)):

            pos = recorder.particles_history[i]
            gbest = recorder.gbest_history[i]

            z_particles = np.array([
                func(np.array(p)) for p in pos
            ])

            frames.append(
                go.Frame(
                    data=[

                        go.Surface(
                            x=X,
                            y=Y,
                            z=Z,
                            colorscale="Viridis",
                            showscale=False,
                            opacity=0.9
                        ),

                        go.Scatter3d(
                            x=pos[:,0],
                            y=pos[:,1],
                            z=z_particles,
                            mode="markers",
                            marker=dict(size=4, color="white")
                        ),

                        go.Scatter3d(
                            x=[gbest[0]],
                            y=[gbest[1]],
                            z=[func(np.array(gbest))],
                            mode="markers",
                            marker=dict(size=10, color="red")
                        )
                    ],

                    traces=[0,1,2]
                )
            )

        fig = go.Figure(
            data=frames[0].data,
            frames=frames
        )

        fig.update_layout(
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Fitness",
                aspectmode="cube"
            ),

            height=750,

            updatemenus=[

                dict(
                    type="buttons",
                    buttons=[

                        dict(
                            label="▶ Play",
                            method="animate",
                            args=[None, {
                                "frame": {"duration": 50, "redraw": False},
                                "transition": {"duration": 0}
                            }]
                        )
                    ]
                )
            ]
        )

        return fig


    st.write("**What is this?** Particles (birds) fly through a mathematical landscape to find the global minimum. The background colors represent height (Dark = Low, Light = High).")

    pop = st.slider("Particles", 10, 50, 30)
    iters = st.slider("Iterations", 20, 150, 80)

    bounds = [-5, 5]

    if st.button("🚀 Run"):
        with st.spinner("Swarm is searching..."):
  
            pso = PSO(
                population=pop,
                max_iter=iters,
                min_bound=bounds[0],
                max_bound=bounds[1]
            )

            recorder = PSORecorder(pso)

            recorder.run()

            fig = build_3d_pso_animation(
                recorder,
                pso.cost_function,
                bounds
            )

            # Retrieve best position and fitness after run
            best_position = recorder.gbest_history[-1]
            best_fitness = min([p.best_fitness for p in pso.swarm])

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"staticPlot": False}
            )
            st.write(f"**Best Position Found:** {best_position}")
            st.write(f"**Lowest Value (Fitness):** {best_fitness:.6f}")
                        