import streamlit as st
import fastf1 as ff1
import fastf1.plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, accuracy_score, silhouette_score
from sklearn.decomposition import PCA
import warnings
import os


st.set_page_config(layout="wide")
session_race = ff1.get_session(2023, 'Italian Grand Prix', 'R') 
session_race.load()

# --- FastF1 Setup ---
try:
    if not os.path.exists('fastf1_cache_ml'):
        os.makedirs('fastf1_cache_ml')
    ff1.Cache.enable_cache('fastf1_cache_ml')
except Exception as e:
    st.error(f"Failed to enable FastF1 cache. Please ensure 'fastf1_cache_ml' directory exists and is writable. Error: {e}")

ff1.plotting.setup_mpl(misc_mpl_mods=False)
warnings.filterwarnings('ignore') # Suppress common warnings, use with caution

# --- Helper Functions ---
@st.cache_data # Cache the data loading function
def load_f1_data(year, gp, session_type):
    session = ff1.get_session(year, gp, session_type)
    try:
        session.load(laps=True, telemetry=False, weather=False, messages=False)
    except Exception as e:
        st.warning(f"Could not load all data for {year} {gp} {session_type}: {e}. Some data might be unavailable.")
        if not hasattr(session, 'laps') or session.laps is None:
            return pd.DataFrame() # Return empty if laps could not be loaded at all

    laps = session.laps.copy() if session.laps is not None else pd.DataFrame()

    if laps.empty:
        return pd.DataFrame()

    # Convert timedelta to seconds
    for col in ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time']:
        if col in laps.columns:
            laps[col + '_sec'] = laps[col].dt.total_seconds()

    # Define columns that are absolutely essential vs. good-to-have
    core_essential_cols = ['Driver', 'LapNumber', 'LapTime', 'Compound', 'Stint'] # For tab2
    other_relevant_cols = ['LapTime_sec', 'Sector1Time_sec', 'Sector2Time_sec',
                           'Sector3Time_sec', 'TyreLife', 'FreshTyre', 'Team',
                           'IsPersonalBest', 'PitOutTime', 'PitInTime']
    
    # Ensure core essential columns are kept if they exist, along with other relevant ones
    combined_cols = list(set(core_essential_cols + other_relevant_cols)) # Unique list
    existing_cols = [col for col in combined_cols if col in laps.columns]
    processed_laps = laps[existing_cols].copy() # Use .copy() here

    # Drop rows only if essential _time_sec_ columns (needed for ML) are missing
    # This ensures original LapTime etc. are preserved even if _sec are NaN for some rows initially
    time_sec_cols_for_ml_na_drop = ['LapTime_sec', 'Sector1Time_sec', 'Sector2Time_sec', 'Sector3Time_sec']
    cols_to_check_for_na_ml = [col for col in time_sec_cols_for_ml_na_drop if col in processed_laps.columns]
    if cols_to_check_for_na_ml:
        # Before dropping, ensure that if a _sec col is missing, its original is also mostly missing, or handle differently
        # For simplicity now, we only drop if these specific _sec columns are NaN
        processed_laps.dropna(subset=cols_to_check_for_na_ml, inplace=True)
        
    return processed_laps

def get_stints(laps_data):
    # Check if necessary columns exist before proceeding
    if laps_data is None or laps_data.empty or 'Driver' not in laps_data.columns:
        return pd.DataFrame()
    
    # Ensure 'Stint' column exists, if not, try to create it based on pit stops
    # Make a copy to avoid SettingWithCopyWarning if modifying laps_data
    data_for_stints = laps_data.copy()
    if 'Stint' not in data_for_stints.columns:
        if 'PitOutTime' in data_for_stints.columns:
            data_for_stints['Stint'] = data_for_stints.groupby('Driver')['PitOutTime'].notna().cumsum() + 1
        else: # If no PitOutTime, cannot reliably create Stint column here
            return pd.DataFrame() # Or, if LapNumber is always present, treat each sequence of laps as a stint if no pit info
    
    # Now 'Stint' should exist if possible, or we returned an empty DF
    if 'Stint' not in data_for_stints.columns:
         return pd.DataFrame()


    stints_grouped = data_for_stints.groupby(['Driver', 'Stint'])
    stint_info = []
    for name, group in stints_grouped:
        if group.empty or 'Compound' not in group.columns or 'LapNumber' not in group.columns: # Check for compound and lapnumber in group
            continue
        stint_info.append({
            'Driver': name[0],
            'Stint': int(name[1]),
            'Compound': group['Compound'].iloc[0] if not group['Compound'].empty else 'UNKNOWN',
            'StartLap': group['LapNumber'].min(),
            'EndLap': group['LapNumber'].max(),
            'LapsInStint': len(group)
        })
    return pd.DataFrame(stint_info)

def preprocess_data_for_ml(df, target_col, features, classification=False, threshold=None, scale_features=True):
    df_processed = df.copy()
    
    if 'TyreLife' in features:
        if 'TyreLife' in df_processed.columns and df_processed['TyreLife'].isnull().any():
            df_processed['TyreLife'] = df_processed['TyreLife'].fillna(df_processed['TyreLife'].median() if not df_processed['TyreLife'].empty else 0)
        elif 'TyreLife' not in df_processed.columns:
            df_processed['TyreLife'] = 0 

    le_compound = None
    if 'Compound' in features:
        if 'Compound' in df_processed.columns:
            df_processed['Compound'] = df_processed['Compound'].astype(str) # Ensure string type before encoding
            le_compound = LabelEncoder()
            # Handle unseen labels during transform by fitting on all known original compounds if possible
            # For simplicity, fit on current data; ideally, fit on a broader set of possible compounds
            df_processed['Compound'] = le_compound.fit_transform(df_processed['Compound'])
        else: 
            df_processed['Compound'] = 0 
            # We still need the 'Compound' column for the feature list if it was expected
            # The features list passed to the model should only contain existing columns
            # This dummy 'Compound' might not be ideal if the model truly expects varied compound data

    # Use only features that actually exist in df_processed at this point
    final_features = [f for f in features if f in df_processed.columns]
    if not final_features:
        st.error("No valid features remaining for ML after preprocessing.")
        return None, None, None, None, None, None

    X = df_processed[final_features]
    y = df_processed[target_col] if target_col in df_processed.columns else None

    if classification:
        if y is None or threshold is None:
             st.error("Target column or threshold missing for classification.")
             return None, None, None, None, None, None
        if y.isnull().any(): # Ensure target for classification has no NaNs
            st.warning("Target column for classification contains NaNs. These rows will be dropped.")
            valid_y_indices = y.dropna().index
            X = X.loc[valid_y_indices]
            y = y.loc[valid_y_indices]
            if X.empty or y.empty:
                st.error("No data left after dropping NaNs from target for classification.")
                return None, None, None, None, None, None
        y = (y < threshold).astype(int)

    if y is None and not classification:
        if X.empty:
            return None, None, None, None, None, None
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
        y_train, y_test = None, None
    elif y is not None:
        if X.empty or y.empty or len(X) != len(y):
            st.error("Feature and target data mismatch or empty after NaNs handling.")
            return None, None, None, None, None, None
        if len(X) < 2 or (classification and len(np.unique(y)) < 2 and len(y) > 0): # Need at least 2 samples, and 2 classes for classification
            st.warning("Not enough samples or class diversity for train/test split or model training.")
            # Fallback: use all data for training, no test set (not ideal)
            X_train, X_test, y_train, y_test = X, pd.DataFrame(columns=X.columns), y, pd.Series(dtype=y.dtype)

        else:
             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if classification and len(y)>1 else None)
    else:
        return None, None, None, None, None, None
    
    scaler = None
    if scale_features and not X_train.empty:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        if not X_test.empty:
            X_test = scaler.transform(X_test)
        
    return X_train, X_test, y_train, y_test, scaler, le_compound


# --- Streamlit App UI ---
st.title("F1 Analysis Dashboard with ML Insights 🏎️💨")

# --- User Inputs ---
st.sidebar.header("Session Selection")
year = st.sidebar.number_input("Year", min_value=2018, max_value=2025, value=2023, key="main_year_input")
events_race = []
try:
    event_schedule = ff1.events.get_event_schedule(year)
    if not event_schedule.empty:
        events_race = event_schedule['EventName'].tolist()
    else:
        st.sidebar.warning(f"No events found for {year}.")
except Exception as e:
    st.sidebar.error(f"Could not fetch event schedule for {year}: {e}")

if not events_race:
    st.sidebar.info("Please select a year with available events or check network connection.")
    st.stop()

default_gp_index = 0
common_gps = ["Italian Grand Prix", "Monza", "British Grand Prix", "Silverstone", "Monaco Grand Prix"]
for gp_name in common_gps:
    if gp_name in events_race:
        default_gp_index = events_race.index(gp_name)
        break
event_name_race = st.sidebar.selectbox("Event Name", events_race, index=default_gp_index, key="main_event_selector")
session_type = st.sidebar.selectbox("Session Type", ["R", "Q", "FP1", "FP2", "FP3"], index=0, key="main_session_selector")


# --- Data Loading & Caching ---
laps_data_global = None
if event_name_race:
    try:
        laps_data_global = load_f1_data(year, event_name_race, session_type)
        if laps_data_global.empty:
            st.warning(f"No lap data found or processed for {event_name_race} {year} - {session_type}.")
            st.stop()
    except Exception as e:
        st.error(f"Error loading F1 data: {e}. Common issues: event name mismatch, no data for selected session/year, or internet connectivity.")
        st.info("FastF1 might not have data for very recent races or some historical FP sessions.")
        st.stop()
else:
    st.info("Please select an event.")
    st.stop()

st.header(f"Analysis: {event_name_race} {year} - {session_type}")
if st.sidebar.checkbox("Show Raw Lap Data", False, key="show_raw_data_checkbox"):
    st.subheader("Raw Lap Data (first 100 rows)")
    st.dataframe(laps_data_global.head(100))

# Tabs for different analyses
tab1, tab2, tab3_ml = st.tabs(["📊 Stint Analysis", "⏱️ Lap Time Performance", "🤖 ML Insights"])

with tab1:
    st.subheader("Driver Stint Overview")
    # get_stints will try to create 'Stint' if not present, or return empty df
    stints_df = get_stints(laps_data_global.copy()) # Pass a copy to get_stints if it modifies the df

    if not stints_df.empty:
        compound_colors_plt = ff1.plotting.get_compound_mapping(session_race).copy()
        compound_colors_plt.update({'UNKNOWN': 'grey'})

        edge_colors_plt = {compound: 'black' for compound in compound_colors_plt}
        drivers_list = sorted(stints_df['Driver'].unique())

        fig_stints, ax_stints = plt.subplots(figsize=(15, max(8, len(drivers_list) * 0.5)))
        legend_elements = {} 
        for i, driver_code in enumerate(drivers_list):
            driver_stints = stints_df[stints_df['Driver'] == driver_code]
            for _, stint_row in driver_stints.iterrows():
                start_lap = stint_row['StartLap']
                end_lap = stint_row['EndLap']
                compound = stint_row['Compound']
                color = compound_colors_plt.get(compound, 'grey')
                
                bar = ax_stints.barh(
                    y=driver_code, width=end_lap - start_lap + 1, left=start_lap,
                    color=color, edgecolor=edge_colors_plt.get(compound, 'black')
                )
                if compound not in legend_elements:
                     legend_elements[compound] = bar[0]

                text_color = 'white' if compound in ['SOFT', 'INTERMEDIATE', 'WET', 'ULTRASOFT', 'HYPERSOFT'] or color in ['#FF3131','#39B54A','#3772FF', '#DA00FE', '#FF00FE'] else 'black'
                ax_stints.text(start_lap + (end_lap - start_lap + 1) / 2, driver_code,
                               compound[:1] if compound != 'UNKNOWN' else '?',
                               ha='center', va='center', fontsize=8, color=text_color)
        ax_stints.set_xlabel("Lap Number")
        ax_stints.set_ylabel("Driver")
        ax_stints.set_title("Tyre Stints per Driver")
        ax_stints.invert_yaxis()
        
        if legend_elements:
             ax_stints.legend(legend_elements.values(), legend_elements.keys(), title="Compounds", bbox_to_anchor=(1.02, 1), loc='upper left')

        plt.tight_layout(rect=[0, 0, 0.88, 1])
        st.pyplot(fig_stints)
    else:
        st.info("Stint overview data could not be processed. Necessary columns ('Driver', 'Stint', 'Compound', 'LapNumber', 'PitOutTime') might be missing or data insufficient.")


# (Keep all your existing imports and helper functions as they are)
# ... (FastF1 Setup, load_f1_data, get_stints, preprocess_data_for_ml) ...
# ... (Streamlit UI, User Inputs, Data Loading, tab1) ...

with tab2:
    st.subheader("Lap Time Performance on Stint")
    
    if laps_data_global is None or laps_data_global.empty:
        st.warning("No lap data has been loaded. Please select a valid session.")
    else:
        # Make a copy for tab-specific modifications to avoid altering global data
        laps_data_tab2 = laps_data_global.copy()

        # --- Essential columns for this specific tab's analysis ---
        essential_cols_tab2_base = ['Driver', 'LapNumber', 'LapTime', 'Stint', 'Compound']
        missing_essential_cols = [col for col in essential_cols_tab2_base if col not in laps_data_tab2.columns]

        if missing_essential_cols:
            st.error(f"The loaded data is missing essential columns for detailed lap time/stint analysis: {missing_essential_cols}. This can happen with some older data or certain session types (e.g., some practice sessions).")
            st.info("Try selecting a different session type (e.g., 'R' or 'Q') or a more recent year if available.")
        else:
            # Ensure LapTime_sec exists, derive if necessary
            if 'LapTime_sec' not in laps_data_tab2.columns and 'LapTime' in laps_data_tab2.columns:
                laps_data_tab2['LapTime_sec'] = laps_data_tab2['LapTime'].dt.total_seconds()
            
            # Now check if LapTime_sec is present after attempting derivation
            essential_cols_for_plot = ['Driver', 'LapNumber', 'LapTime_sec', 'Stint', 'Compound']
            if not all(col in laps_data_tab2.columns for col in essential_cols_for_plot):
                missing_plot_cols = [col for col in essential_cols_for_plot if col not in laps_data_tab2.columns]
                st.error(f"Critical columns for plotting are missing after processing: {missing_plot_cols}. 'LapTime_sec' could not be derived if 'LapTime' was missing.")
                st.stop()


            drivers_laps_list = pd.unique(laps_data_tab2['Driver']).tolist()
            if not drivers_laps_list:
                st.warning("No drivers found in the processed lap data.")
            else:
                selected_driver_stint_analysis = st.selectbox(
                    "Select Driver ", 
                    drivers_laps_list, 
                    index=0, 
                    key="tab2_driver_select"
                )
                
                driver_laps_data_analysis = laps_data_tab2[laps_data_tab2['Driver'] == selected_driver_stint_analysis]
                
                # Columns needed for this part of tab2's stint processing
                cols_for_this_analysis_driver = ['LapTime_sec', 'LapNumber', 'Stint', 'Compound']
                if not all(col in driver_laps_data_analysis.columns for col in cols_for_this_analysis_driver):
                    st.warning(f"Driver {selected_driver_stint_analysis} is missing some data needed for full stint analysis: {cols_for_this_analysis_driver}")
                else:
                    driver_laps_data_analysis = driver_laps_data_analysis.dropna(
                        subset=cols_for_this_analysis_driver
                    )

                    if driver_laps_data_analysis.empty:
                        st.warning(f"No complete lap time data (after cleaning NaNs) for driver {selected_driver_stint_analysis}.")
                    else:
                        try:
                            driver_laps_data_analysis['Stint'] = driver_laps_data_analysis['Stint'].astype(int)
                        except ValueError:
                            st.error("Could not convert 'Stint' column to integer for the selected driver. Data might be inconsistent.")
                            st.stop()
                            
                        stint_numbers_analysis = sorted(driver_laps_data_analysis['Stint'].unique())
                        
                        if not stint_numbers_analysis:
                            st.warning(f"No valid stints found for driver {selected_driver_stint_analysis} after processing.")
                        else:
                            selected_stint_num_analysis = st.selectbox(
                                "Select Stint Number ", 
                                stint_numbers_analysis, 
                                key="tab2_stint_select"
                            )
                            # Now, stint_laps_analysis will definitely have LapTime_sec if driver_laps_data_analysis had it
                            stint_laps_analysis = driver_laps_data_analysis[driver_laps_data_analysis['Stint'] == selected_stint_num_analysis].copy()
                            
                            if stint_laps_analysis.empty:
                                st.info(f"No data for Stint {selected_stint_num_analysis} for driver {selected_driver_stint_analysis}.")
                            else:
                                compound_used_analysis = stint_laps_analysis['Compound'].iloc[0]
                                min_lap_stint = int(stint_laps_analysis['LapNumber'].min())
                                max_lap_stint = int(stint_laps_analysis['LapNumber'].max())
                                animated_lap_limit = max_lap_stint 
                                if min_lap_stint < max_lap_stint :
                                    animated_lap_limit = st.slider("Show laps up to:", min_lap_stint, max_lap_stint, max_lap_stint, 1, key="tab2_lap_slider")
                                
                                # This is where laps_to_plot_plotly gets LapTime_sec from stint_laps_analysis
                                laps_to_plot_plotly = stint_laps_analysis[stint_laps_analysis['LapNumber'] <= animated_lap_limit]

                                if not laps_to_plot_plotly.empty:
                                    fig_lap_times_plotly = go.Figure()
                                    plotly_compound_color = compound_colors_plt.get(compound_used_analysis, 'grey')
                                    fig_lap_times_plotly.add_trace(go.Scatter(
                                        x=laps_to_plot_plotly['LapNumber'], 
                                        y=laps_to_plot_plotly['LapTime_sec'], # Standardized to LapTime_sec
                                        mode='lines+markers', name=f'{compound_used_analysis} Lap Times',
                                        marker=dict(color=plotly_compound_color), line=dict(color=plotly_compound_color)
                                    ))
                                    x_vals_plotly = laps_to_plot_plotly['LapNumber'].values
                                    y_vals_plotly = laps_to_plot_plotly['LapTime_sec'].values # Standardized
                                    if len(x_vals_plotly) >= 2:
                                        try:
                                            trend_coeffs_plotly = np.polyfit(x_vals_plotly, y_vals_plotly, 1)
                                            trend_line_func_plotly = np.poly1d(trend_coeffs_plotly)
                                            fig_lap_times_plotly.add_trace(go.Scatter(
                                                x=x_vals_plotly, y=trend_line_func_plotly(x_vals_plotly),
                                                mode='lines', name=f"Trend (Deg: {trend_coeffs_plotly[0]:.3f} s/lap)",
                                                line=dict(color='rgba(255,0,0,0.6)', dash='dash')
                                            ))
                                        except np.linalg.LinAlgError: st.caption("Could not compute trend line.")
                                    
                                    min_time_overall_stint = stint_laps_analysis['LapTime_sec'].min() - 0.5 if not stint_laps_analysis.empty else 0
                                    max_time_overall_stint = stint_laps_analysis['LapTime_sec'].max() + 0.5 if not stint_laps_analysis.empty else 100
                                    
                                    fig_lap_times_plotly.update_layout(
                                        xaxis_title="Lap Number", yaxis_title="Lap Time (seconds)",
                                        title=f"{selected_driver_stint_analysis} - Stint {selected_stint_num_analysis} ({compound_used_analysis})",
                                        yaxis_range=[min_time_overall_stint, max_time_overall_stint],
                                        xaxis_range=[min_lap_stint -1 if min_lap_stint else 0, max_lap_stint + 1 if max_lap_stint else 10]
                                    )
                                    st.plotly_chart(fig_lap_times_plotly, use_container_width=True)
                                else: 
                                    st.info("No laps to plot for selected range.")

# --- ML Insights Tab ---
with tab3_ml:
    st.subheader("Machine Learning Insights")
    st.write("Models are trained on the currently loaded session data. Predictions are indicative for this session.")

    ml_base_df = laps_data_global.copy()
    
    # Ensure necessary _sec columns are present for ML features
    # (load_f1_data should have already created them if original columns were present)
    sec_cols_needed = ['LapTime_sec', 'Sector1Time_sec', 'Sector2Time_sec', 'Sector3Time_sec']
    if not all(col in ml_base_df.columns for col in sec_cols_needed):
        st.warning(f"One or more time_sec columns ({sec_cols_needed}) are missing. ML features might be limited.")
        # Attempt to derive them again if original timedelta columns exist
        for col_base in ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time']:
            if col_base in ml_base_df.columns and f"{col_base}_sec" not in ml_base_df.columns:
                ml_base_df[f"{col_base}_sec"] = ml_base_df[col_base].dt.total_seconds()


    # --- 1. Lap Time Prediction (Linear Regression) ---
    st.markdown("---")
    st.markdown("#### ⏱️ Lap Time Prediction (Linear Regression)")
    
    features_reg = ['Sector1Time_sec', 'Sector2Time_sec', 'Sector3Time_sec', 'TyreLife', 'Compound']
    target_reg = 'LapTime_sec'
    
    required_cols_for_lr = [target_reg, 'Sector1Time_sec', 'Sector2Time_sec', 'Sector3Time_sec'] # Compound & TyreLife handled by preprocess
    if all(col in ml_base_df.columns for col in required_cols_for_lr):
        ml_df_reg = ml_base_df.copy()
        
        X_train_reg, X_test_reg, y_train_reg, y_test_reg, scaler_reg, le_compound_reg = preprocess_data_for_ml(
            ml_df_reg, target_reg, features_reg, scale_features=True
        )

        if X_train_reg is not None and y_train_reg is not None and len(X_train_reg) > 0:
            model_lr = LinearRegression()
            model_lr.fit(X_train_reg, y_train_reg)
            if X_test_reg is not None and y_test_reg is not None and len(X_test_reg) > 0:
                y_pred_reg = model_lr.predict(X_test_reg)
                r2 = r2_score(y_test_reg, y_pred_reg)
                st.write(f"Linear Regression Model R² score (on test set): {r2:.4f}")
            else:
                st.write("Linear Regression Model trained. Not enough test data for R² score.")
            st.caption("Predicts lap time based on sector times, tyre life, and compound.")

            with st.expander("Predict Lap Time for a New Lap"):
                s1 = st.number_input("Sector 1 Time (s)", value=float(ml_df_reg['Sector1Time_sec'].median() if not ml_df_reg.empty and 'Sector1Time_sec' in ml_df_reg else 25.0), step=0.1, key="lr_s1_input")
                s2 = st.number_input("Sector 2 Time (s)", value=float(ml_df_reg['Sector2Time_sec'].median() if not ml_df_reg.empty and 'Sector2Time_sec' in ml_df_reg else 30.0), step=0.1, key="lr_s2_input")
                s3 = st.number_input("Sector 3 Time (s)", value=float(ml_df_reg['Sector3Time_sec'].median() if not ml_df_reg.empty and 'Sector3Time_sec' in ml_df_reg else 28.0), step=0.1, key="lr_s3_input")
                tyre_life_input = st.number_input("Tyre Life (laps)", value=int(ml_df_reg['TyreLife'].median() if 'TyreLife' in ml_df_reg and not ml_df_reg.empty else 5), min_value=0, step=1, key="lr_tyre_input")
                
                original_compounds_reg = ["UNKNOWN"]
                if le_compound_reg:
                    original_compounds_reg = le_compound_reg.classes_.tolist()
                elif 'Compound' in ml_df_reg.columns: # Fallback if le_compound_reg is None but Compound column exists
                     original_compounds_reg = ml_df_reg['Compound'].astype(str).unique().tolist()
                
                compound_input_reg_str = st.selectbox("Tyre Compound (for LR)", original_compounds_reg, key="lr_compound_select_input")

                if st.button("Predict Lap Time", key="lr_predict_btn_input"):
                    compound_encoded_reg = 0 # Default if not found or encoder not available
                    if le_compound_reg and compound_input_reg_str in le_compound_reg.classes_:
                         compound_encoded_reg = le_compound_reg.transform([compound_input_reg_str])[0]
                    
                    predict_df_columns = [f for f in features_reg if f in (scaler_reg.feature_names_in_ if scaler_reg and hasattr(scaler_reg, 'feature_names_in_') else ml_df_reg.columns)]
                    if not predict_df_columns:
                        st.error("Could not determine feature columns for prediction.")
                    else:
                        input_data_dict = {'Sector1Time_sec': s1, 'Sector2Time_sec': s2, 'Sector3Time_sec': s3, 'TyreLife': tyre_life_input, 'Compound': compound_encoded_reg}
                        current_input_features = [input_data_dict[f] for f in predict_df_columns]

                        input_features_df = pd.DataFrame([current_input_features], columns=predict_df_columns)
                        
                        input_features_scaled = scaler_reg.transform(input_features_df) if scaler_reg else input_features_df
                        predicted_time = model_lr.predict(input_features_scaled)
                        st.success(f"Predicted Lap Time: {predicted_time[0]:.3f} seconds")
        else:
            st.warning("Not enough data or valid features for Linear Regression.")
    else:
        st.warning("Required time features for Lap Time Prediction are missing.")


    # --- 2. Fast Lap Classification (SVM) ---
    st.markdown("---")
    st.markdown("#### 🏎️ Fast Lap Classification (SVM)")
    
    features_clf = ['Sector1Time_sec', 'Sector2Time_sec', 'Sector3Time_sec', 'TyreLife', 'Compound']
    target_clf = 'LapTime_sec'

    if all(col in ml_base_df.columns for col in required_cols_for_lr): # Check for same base cols as regression
        ml_df_clf = ml_base_df.copy()
        
        if not ml_df_clf.empty and target_clf in ml_df_clf.columns and len(ml_df_clf) > 5:
            threshold_fast_lap = ml_df_clf[target_clf].median()
            st.write(f"Fast Lap Threshold (Median Lap Time): {threshold_fast_lap:.3f} seconds")

            X_train_clf, X_test_clf, y_train_clf, y_test_clf, scaler_clf, le_compound_clf = preprocess_data_for_ml(
                ml_df_clf, target_clf, features_clf, classification=True, threshold=threshold_fast_lap, scale_features=True
            )

            if X_train_clf is not None and y_train_clf is not None and len(y_train_clf) > 0 and len(np.unique(y_train_clf)) > 1 :
                model_svm = SVC(kernel='linear', probability=True)
                model_svm.fit(X_train_clf, y_train_clf)
                if X_test_clf is not None and y_test_clf is not None and len(X_test_clf) > 0:
                    y_pred_svm = model_svm.predict(X_test_clf)
                    accuracy_svm = accuracy_score(y_test_clf, y_pred_svm)
                    st.write(f"SVM Model Accuracy (on test set): {accuracy_svm:.4f}")
                else:
                    st.write("SVM Model trained. Not enough test data for accuracy score.")
                st.caption("Classifies if a lap is 'fast' (1, below median) or 'not fast' (0).")

                with st.expander("Classify a Lap (SVM)"):
                    s1_svm = st.number_input("Sector 1 Time (s)", value=float(ml_df_clf['Sector1Time_sec'].median() if not ml_df_clf.empty else 25.0), step=0.1, key="svm_s1_input_exp")
                    s2_svm = st.number_input("Sector 2 Time (s)", value=float(ml_df_clf['Sector2Time_sec'].median() if not ml_df_clf.empty else 30.0), step=0.1, key="svm_s2_input_exp")
                    s3_svm = st.number_input("Sector 3 Time (s)", value=float(ml_df_clf['Sector3Time_sec'].median() if not ml_df_clf.empty else 28.0), step=0.1, key="svm_s3_input_exp")
                    tyre_life_svm = st.number_input("Tyre Life (laps)", value=int(ml_df_clf['TyreLife'].median() if 'TyreLife' in ml_df_clf and not ml_df_clf.empty else 5), min_value=0, step=1, key="svm_tyre_input_exp")
                    
                    original_compounds_svm = le_compound_clf.classes_ if le_compound_clf else ml_df_clf['Compound'].astype(str).unique().tolist() if 'Compound' in ml_df_clf else ["UNKNOWN"]
                    compound_input_svm_str = st.selectbox("Tyre Compound (for SVM)", original_compounds_svm, key="svm_compound_select_exp")

                    if st.button("Classify Lap", key="svm_classify_btn_exp"):
                        compound_encoded_svm = 0
                        if le_compound_clf and compound_input_svm_str in le_compound_clf.classes_:
                            compound_encoded_svm = le_compound_clf.transform([compound_input_svm_str])[0]
                        
                        predict_df_columns_clf = [f for f in features_clf if f in (scaler_clf.feature_names_in_ if scaler_clf and hasattr(scaler_clf, 'feature_names_in_') else ml_df_clf.columns)]
                        if not predict_df_columns_clf:
                            st.error("Could not determine feature columns for SVM prediction.")
                        else:
                            input_data_dict_svm = {'Sector1Time_sec': s1_svm, 'Sector2Time_sec': s2_svm, 'Sector3Time_sec': s3_svm, 'TyreLife': tyre_life_svm, 'Compound': compound_encoded_svm}
                            current_input_features_svm = [input_data_dict_svm[f] for f in predict_df_columns_clf]
                            input_features_svm_df = pd.DataFrame([current_input_features_svm], columns=predict_df_columns_clf)
                        
                            input_features_svm_scaled = scaler_clf.transform(input_features_svm_df) if scaler_clf else input_features_svm_df
                            prediction_svm = model_svm.predict(input_features_svm_scaled)[0]
                            prediction_proba_svm = model_svm.predict_proba(input_features_svm_scaled)[0]
                            result_text = "Fast Lap 🎉" if prediction_svm == 1 else "Not a Fast Lap"
                            st.success(f"Predicted Classification: {result_text} (Confidence for 'Fast': {prediction_proba_svm[1]:.2f})")
            else:
                st.warning("Not enough data, features, or class variation for SVM.")
        else:
            st.warning("DataFrame for SVM classification is empty or too small.")
    else:
        st.warning("Required time features for Fast Lap Classification are missing.")

    # --- 3. Lap Clustering (K-Means) ---
    st.markdown("---")
    st.markdown("#### 📊 Lap Clustering (K-Means)")
    
    features_clust_list = ['LapTime_sec', 'Sector1Time_sec', 'Sector2Time_sec', 'Sector3Time_sec', 'TyreLife']
    if all(f_sec in ml_base_df.columns for f_sec in ['LapTime_sec','Sector1Time_sec', 'Sector2Time_sec', 'Sector3Time_sec']):
        ml_df_clust = ml_base_df.copy()
        if 'TyreLife' not in ml_df_clust.columns: ml_df_clust['TyreLife'] = 0
        
        final_features_clust = [f for f in features_clust_list if f in ml_df_clust.columns]
        ml_df_clust = ml_df_clust.dropna(subset=final_features_clust)

        if not ml_df_clust.empty and len(ml_df_clust) > 3 and final_features_clust:
            X_clust_scaled = StandardScaler().fit_transform(ml_df_clust[final_features_clust])
            
            k_clusters = st.slider("Number of Clusters (K)", 2, 10, 3, key="kmeans_k_slider")
            model_kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init='auto')
            clusters = model_kmeans.fit_predict(X_clust_scaled)
            
            if len(X_clust_scaled) > 1 and len(np.unique(clusters)) > 1:
                score_silhouette = silhouette_score(X_clust_scaled, clusters)
                st.write(f"K-Means Silhouette Score: {score_silhouette:.4f}")
            else:
                st.write("K-Means executed. Silhouette Score requires >1 cluster & >1 sample per cluster.")

            st.caption("Clusters laps based on their times and tyre life. Visualize with PCA:")
            
            if X_clust_scaled.shape[1] >=2 and X_clust_scaled.shape[0] >=2 :
                pca = PCA(n_components=2)
                X_clust_pca = pca.fit_transform(X_clust_scaled)
                
                df_pca_clust = pd.DataFrame(X_clust_pca, columns=['PC1', 'PC2'])
                df_pca_clust['Cluster'] = clusters.astype(str)
                df_pca_clust['Driver'] = ml_df_clust['Driver'].values if 'Driver' in ml_df_clust.columns else 'N/A'
                df_pca_clust['LapTime_sec'] = ml_df_clust['LapTime_sec'].values

                fig_kmeans = px.scatter(df_pca_clust, x='PC1', y='PC2', color='Cluster',
                                        hover_data=['Driver', 'LapTime_sec'],
                                        title=f'Lap Clusters (PCA Reduced - K={k_clusters})',
                                        color_discrete_map={str(i): px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] for i in range(k_clusters)})
                st.plotly_chart(fig_kmeans, use_container_width=True)
            else:
                st.warning("Not enough features or samples for PCA visualization of clusters.")
        else:
            st.warning("Not enough data or valid features to perform K-Means clustering.")
    else:
        st.warning("Required time features for Lap Clustering are missing.")
