import streamlit as st
from sqlalchemy import create_engine
import uuid
from datetime import datetime, timezone, timedelta
import pandas as pd
import plotly.graph_objects as go

from config import Settings
from queries import get_baseline_data, get_info_for_active_projects, get_target_data, Target, BaselineData
from dataclasses import dataclass

@dataclass
class Cycle:
    """Represents a measurement cycle with its targets."""
    timestamp: int
    targets: list[Target]

def create_sidebar() -> None:
    """Create sidebar for data loading options."""
    
    with st.sidebar:
        
        # Data loading method selection
        loading_method = st.radio(
            "Select data loading method",
            ["Project + Device", "Stream ID"]
        )
        
        # Stream ID or Project/Device inputs
        stream_id = None
        selected_project_device = None
        
        if loading_method == "Stream ID":
            stream_id_input = st.text_input(
                "Stream ID",
                placeholder="Enter UUID"
            )
            
            # Validate UUID
            if stream_id_input:
                try:
                    stream_id = uuid.UUID(stream_id_input)
                except ValueError:
                    st.error("Invalid UUID format")
                    stream_id = None
                    
        else:  # Project + Device
            try:
                # Get project/device info
                platform_engine = st.session_state['platform_engine']
                project_devices = get_info_for_active_projects(platform_engine)
                
                if project_devices:
                    # Extract unique projects
                    unique_projects = list(set(pd.project_name for pd in project_devices))
                    unique_projects.sort()
                    
                    # Project dropdown
                    selected_project = st.selectbox(
                        "Select Project",
                        unique_projects,
                        index=0
                    )
                    
                    if selected_project:
                        # Filter devices for selected project
                        project_devices_filtered = [
                            pd for pd in project_devices 
                            if pd.project_name == selected_project
                        ]
                        
                        device_names = [pd.device_name for pd in project_devices_filtered]
                        
                        # Device dropdown
                        selected_device = st.selectbox(
                            "Select Device",
                            device_names,
                            index=0
                        )
                        
                        if selected_device:
                            # Find the corresponding ProjectStreamInfo
                            selected_project_device = next(
                                pd for pd in project_devices_filtered 
                                if pd.device_name == selected_device
                            )
                            stream_id = selected_project_device.stream_id
                        
                else:
                    st.error("No active projects found")
                    
            except Exception as e:
                st.error(f"Error loading projects: {str(e)}")
        
        # Date picker
        selected_date = st.date_input(
            "Select date:",
            value=datetime.now(timezone.utc).date()
        )
        
        # Load Data button
        if st.button("Load Data", type="primary"):
            if stream_id and selected_date:
                # Convert date to UTC timestamps (start and end of day)
                start_datetime = datetime.combine(selected_date, datetime.min.time()).replace(tzinfo=timezone.utc)
                end_datetime = start_datetime + timedelta(days=1) - timedelta(microseconds=1)
                
                start_timestamp = int(start_datetime.timestamp())
                end_timestamp = int(end_datetime.timestamp())
                
                # Store in session state for main app to use
                st.session_state['stream_id'] = stream_id
                st.session_state['start_timestamp'] = start_timestamp
                st.session_state['end_timestamp'] = end_timestamp
                st.session_state['selected_date'] = selected_date
                st.session_state['load_timestamp'] = datetime.now().timestamp()  # Track when data was loaded
                
                if loading_method == "Project + Device" and selected_project_device:
                    st.session_state['project_info'] = selected_project_device
                
                # Load target data automatically
                load_target_data(stream_id)
                # Update the page with new data
                st.rerun()
                
            else:
                if not stream_id:
                    st.error("Please provide a valid stream ID")
                if not selected_date:
                    st.error("Please select a date")


def load_target_data(stream_id: uuid.UUID) -> list[Target]:
    """
    Load target data for the given stream_id using pipeline engine from session state.
    
    Args:
        stream_id: UUID of the stream to load data for
    
    Returns:
        List of Target objects containing target data
    """
    try:
        pipeline_engine = st.session_state['pipeline_engine']
        targets = get_target_data(
            pipeline_engine, 
            stream_id, 
            st.session_state['start_timestamp'], 
            st.session_state['end_timestamp']
        )
        
        # Convert targets to cycles
        cycles = convert_targets_to_cycles(targets)
        
        # Store cycles in session state for other components to use
        st.session_state['cycles_data'] = cycles
        
        return targets
        
    except Exception as e:
        st.error(f"Error loading target data: {str(e)}")
        return []


def convert_targets_to_cycles(targets: list[Target]) -> list[Cycle]:
    """
    Convert list of targets to list of cycles grouped by timestamp.
    
    Args:
        targets: List of Target objects
        
    Returns:
        List of Cycle objects, one per unique timestamp
    """
    # Group targets by timestamp
    cycles_dict: dict[int, list[Target]] = {}
    for target in targets:
        if target.timestamp not in cycles_dict:
            cycles_dict[target.timestamp] = []
        cycles_dict[target.timestamp].append(target)
    
    # Create Cycle objects
    cycles = []
    for timestamp, cycle_targets in cycles_dict.items():
        cycles.append(Cycle(timestamp=timestamp, targets=cycle_targets))
    
    # Sort by timestamp
    cycles.sort(key=lambda c: c.timestamp)
    
    return cycles


def keep_baseline_matching_targets(cycles: list[Cycle], baseline_data: BaselineData) -> list[Cycle]:
    """
    Remove targets that don't match baseline targets by name.
    
    Args:
        cycles: List of cycles to filter
        baseline_data: Baseline data containing target names to match against
        
    Returns:
        List of cycles with only baseline-matching targets
    """
    baseline_target_names = set(target.name for target in baseline_data.targets)
    
    filtered_cycles = []
    for cycle in cycles:
        matching_targets = [
            target for target in cycle.targets 
            if target.name in baseline_target_names
        ]
        
        if matching_targets:  # Only keep cycles that have at least one matching target
            filtered_cycle = Cycle(timestamp=cycle.timestamp, targets=matching_targets)
            filtered_cycles.append(filtered_cycle)
    
    return filtered_cycles


def filter_by_detection_count(cycles: list[Cycle], expected_count: int) -> list[Cycle]:
    """
    Filter cycles based on minimum number of target detections.
    
    Args:
        cycles: List of cycles to filter
        expected_count: Minimum number of targets that should be detected
        
    Returns:
        List of cycles with at least expected_count targets
    """
    return [cycle for cycle in cycles if len(cycle.targets) >= expected_count]


def filter_by_cqf(cycles: list[Cycle], cqf_threshold: int) -> list[Cycle]:
    """
    Filter cycles where ALL targets pass the CQF threshold.
    
    Args:
        cycles: List of cycles to filter
        cqf_threshold: Minimum CQF value for all targets
        
    Returns:
        List of cycles where all targets meet CQF threshold
    """
    return [
        cycle for cycle in cycles 
        if all(target.cqf >= cqf_threshold for target in cycle.targets)
    ]


def filter_by_snr(cycles: list[Cycle], snr_threshold: int) -> list[Cycle]:
    """
    Filter cycles where ALL targets pass the SNR threshold.
    
    Args:
        cycles: List of cycles to filter
        snr_threshold: Minimum SNR value for all targets
        
    Returns:
        List of cycles where all targets meet SNR threshold
    """
    return [
        cycle for cycle in cycles 
        if all(target.snr >= snr_threshold for target in cycle.targets)
    ]


def calculate_highest_common_threshold(cycles: list[Cycle], metric: str) -> int:
    """
    Calculate the highest common threshold for a specific metric that all targets across all timestamps can meet.
    
    The method analyzes each timestamp separately, finds the minimum value among targets in that timestamp,
    then returns the maximum of those per-timestamp minimums as the highest common threshold.
    
    Args:
        cycles: List of cycles already filtered to include only baseline-matching targets
        metric: Either "cqf" or "snr" to specify which metric to analyze
        
    Returns:
        Highest common threshold as integer
    """
    if not cycles:
        return 80 if metric == "cqf" else 23  # Default fallback values
    
    if metric not in ["cqf", "snr"]:
        raise ValueError(f"Invalid metric '{metric}'. Must be 'cqf' or 'snr'")
    
    # Calculate minimum value per timestamp
    per_timestamp_minimums = []
    
    for cycle in cycles:
        if not cycle.targets:
            continue
            
        # Get values for the specified metric from all targets in this timestamp
        if metric == "cqf":
            timestamp_values = [target.cqf for target in cycle.targets]
        else:  # metric == "snr"
            timestamp_values = [target.snr for target in cycle.targets]
        
        # Find minimum value for this timestamp
        timestamp_minimum = min(timestamp_values)
        per_timestamp_minimums.append(timestamp_minimum)
    
    if not per_timestamp_minimums:
        return 80 if metric == "cqf" else 23  # Default fallback values
    
    # The highest common threshold is the maximum of per-timestamp minimums
    highest_common_threshold = int(max(per_timestamp_minimums))
    
    return highest_common_threshold


def apply_cycle_filters(cycles: list[Cycle], baseline_data: BaselineData, expected_count: int, cqf_threshold: int, snr_threshold: int) -> list[Cycle]:
    """
    Apply all filtering steps to cycles in sequence.
    
    Args:
        cycles: List of cycles to filter
        baseline_data: Baseline data containing target names to match against
        expected_count: Minimum number of targets that should be detected
        cqf_threshold: Minimum CQF value for all targets
        snr_threshold: Minimum SNR value for all targets
        
    Returns:
        List of cycles that pass all filters
    """
    # Apply filters in sequence
    filtered_cycles = keep_baseline_matching_targets(cycles, baseline_data)
    filtered_cycles = filter_by_detection_count(filtered_cycles, expected_count)
    filtered_cycles = filter_by_cqf(filtered_cycles, cqf_threshold)
    filtered_cycles = filter_by_snr(filtered_cycles, snr_threshold)
    
    return filtered_cycles


def display_filtered_cycles_table(filtered_cycles: list[Cycle]) -> None:
    """Display filtered cycles in a clickable table for timestamp selection."""
    if not filtered_cycles:
        st.warning("No cycles match the current filter criteria.")
        return
    
    st.subheader(f"Best cycle candidates ({len(filtered_cycles)} found)")
    
    # Initialize selected timestamp if not set (default to first cycle)
    if 'selected_timestamp' not in st.session_state and filtered_cycles:
        st.session_state['selected_timestamp'] = filtered_cycles[0].timestamp
    
    # Convert timestamps to datetime strings for display
    cycle_data = []
    for cycle in filtered_cycles:
        dt = datetime.fromtimestamp(cycle.timestamp, tz=timezone.utc)
        cycle_data.append({
            "Timestamp": dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        })
    
    df = pd.DataFrame(cycle_data)
    
    # Display clickable dataframe with row selection
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="timestamp_selection"
    )
    
    # Handle row selection using session state
    if "timestamp_selection" in st.session_state and st.session_state.timestamp_selection["selection"]["rows"]:
        selected_row_index = st.session_state.timestamp_selection["selection"]["rows"][0]
        if 0 <= selected_row_index < len(filtered_cycles):
            new_selected_timestamp = filtered_cycles[selected_row_index].timestamp
            # Update session state if selection changed
            if st.session_state['selected_timestamp'] != new_selected_timestamp:
                st.session_state['selected_timestamp'] = new_selected_timestamp


def display_target_chart(filtered_cycles: list[Cycle], selected_timestamp: int) -> None:
    """Display scatter plot chart for targets in the selected timestamp cycle."""
    # Find the cycle matching the selected timestamp
    selected_cycle = next((cycle for cycle in filtered_cycles if cycle.timestamp == selected_timestamp), None)
    if selected_cycle is None:
        st.warning("No cycle found for the selected timestamp.")
        return
    
    # Convert timestamp to readable format for chart title
    dt = datetime.fromtimestamp(selected_timestamp, tz=timezone.utc)
    chart_title = f"Viewfinder - {dt.strftime('%Y-%m-%d %H:%M:%S UTC')}"
    
    # Prepare data for plotting
    x_values = [target.x_px for target in selected_cycle.targets]
    y_values = [target.y_px for target in selected_cycle.targets]
    
    # Create annotations for each target
    annotations = []
    for target in selected_cycle.targets:
        target_name = target.name
        cqf_val = int(target.cqf)  # No decimal
        snr_val = int(target.snr)  # No decimal
        
        annotation_text = f"{target_name}<br>cqf = {cqf_val}<br>snr = {snr_val}"
        
        annotations.append(
            dict(
                x=target.x_px,
                y=target.y_px,
                text=annotation_text,
                bgcolor="black",
                bordercolor="white",
                borderwidth=1,
                font=dict(size=12)
            )
        )
    
    # Create the scatter plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_values,
        mode='markers',
        marker=dict(
            size=8,
            color='yellow',
            symbol='circle'
        ),
        name='Targets',
        showlegend=False
    ))
    
    # Update layout
    fig.update_layout(
        title=chart_title,
        yaxis=dict(range=[1280, 0]),
        xaxis=dict(range=[0, 1024],
                   scaleanchor="y",
                   scaleratio=1
        ),
        annotations=annotations,
        showlegend=False,
        width=512,  # Explicit width to prevent full-screen stretching
        height=640  # Explicit height to maintain proper aspect ratio
    )
    
    # Display the chart without full container width
    st.plotly_chart(fig, use_container_width=False)


def create_filter_sliders() -> None:
    """Create filter sliders for target data filtering."""
    
    if 'cycles_data' not in st.session_state or not st.session_state['cycles_data']:
        st.warning("No cycles found")
        return
        
    cycles = st.session_state['cycles_data']
    stream_id = st.session_state['stream_id']
    
    # Calculate total count of cycles
    total_cycles = len(cycles)
    
    st.info(f"{total_cycles} cycles found")
    
    # Load baseline data to get max targets count
    try:
        pipeline_engine = st.session_state['pipeline_engine']
        baseline_data = get_baseline_data(pipeline_engine, stream_id)
        max_baseline_targets = len(baseline_data.targets)
    except Exception as e:
        st.error(f"Error loading baseline data: {str(e)}")
        return  # Exit if baseline data can't be loaded
    
    # Compute baseline filtered cycles to avoid duplicate filtering
    baseline_filtered_cycles = keep_baseline_matching_targets(cycles, baseline_data)
    
    # Calculate optimal thresholds for initialization
    # First apply detection filter with max baseline targets
    detection_filtered_cycles = filter_by_detection_count(baseline_filtered_cycles, max_baseline_targets)
    calculated_cqf = calculate_highest_common_threshold(detection_filtered_cycles, "cqf")
    
    # Then apply CQF filter and calculate optimal SNR
    cqf_filtered_for_snr = filter_by_cqf(detection_filtered_cycles, calculated_cqf)
    calculated_snr = calculate_highest_common_threshold(cqf_filtered_for_snr, "snr")
    
    # Check if we have new data loaded and need to re-optimize filters
    current_data_key = f"{stream_id}_{st.session_state.get('start_timestamp', '')}_{st.session_state.get('end_timestamp', '')}_{st.session_state.get('load_timestamp', '')}"
    previous_data_key = st.session_state.get('current_data_key', '')
    
    # Initialize session state values if not present or if new data is loaded
    if current_data_key != previous_data_key:
        # New data loaded - re-optimize filter values
        st.session_state['targets_detected_threshold'] = max_baseline_targets
        st.session_state['cqf_threshold'] = calculated_cqf
        st.session_state['snr_threshold'] = calculated_snr
        st.session_state['current_data_key'] = current_data_key
    else:
        # Same data - preserve existing filter values if they exist
        if 'targets_detected_threshold' not in st.session_state:
            st.session_state['targets_detected_threshold'] = max_baseline_targets
        if 'cqf_threshold' not in st.session_state:
            st.session_state['cqf_threshold'] = calculated_cqf
        if 'snr_threshold' not in st.session_state:
            st.session_state['snr_threshold'] = calculated_snr
    
    st.subheader("Filters")
    
    # Create dynamic widget keys that reset when new data is loaded
    widget_key_suffix = st.session_state.get('current_data_key', 'default')
    
    # Get all input values first
    targets_detected_input = st.text_input(
        f"Expected count of target detections (range: 1-{max_baseline_targets})",
        value=str(st.session_state['targets_detected_threshold']),
        key=f"targets_detected_input_{widget_key_suffix}",
        help="Press Enter to apply"
    )
    
    cqf_threshold_input = st.text_input(
        "CQF threshold [%] (range: 0-100)",
        value=str(st.session_state['cqf_threshold']),
        key=f"cqf_threshold_input_{widget_key_suffix}",
        help="Press Enter to apply"
    )
    
    snr_threshold_input = st.text_input(
        "SNR threshold [dB] (range: 0-40)",
        value=str(st.session_state['snr_threshold']),
        key=f"snr_threshold_input_{widget_key_suffix}",
        help="Press Enter to apply"
    )
    
    # Validate all inputs
    try:
        targets_detected = int(targets_detected_input)
        if 1 <= targets_detected <= max_baseline_targets:
            st.session_state['targets_detected_threshold'] = targets_detected
        else:
            st.error(f"Targets detected must be between 1 and {max_baseline_targets}")
            targets_detected = st.session_state['targets_detected_threshold']
    except ValueError:
        st.error("Please enter a valid integer for targets detected")
        targets_detected = st.session_state['targets_detected_threshold']
    
    try:
        cqf_threshold = int(cqf_threshold_input)
        if 0 <= cqf_threshold <= 100:
            st.session_state['cqf_threshold'] = cqf_threshold
        else:
            st.error("CQF threshold must be between 0 and 100")
            cqf_threshold = st.session_state['cqf_threshold']
    except ValueError:
        st.error("Please enter a valid integer for CQF threshold")
        cqf_threshold = st.session_state['cqf_threshold']
    
    try:
        snr_threshold = int(snr_threshold_input)
        if 0 <= snr_threshold <= 40:
            st.session_state['snr_threshold'] = snr_threshold
        else:
            st.error("SNR threshold must be between 0 and 40")
            snr_threshold = st.session_state['snr_threshold']
    except ValueError:
        st.error("Please enter a valid integer for SNR threshold")
        snr_threshold = st.session_state['snr_threshold']

    
    # Apply filters in sequence
    detection_filtered = filter_by_detection_count(baseline_filtered_cycles, targets_detected)
    cqf_filtered = filter_by_cqf(detection_filtered, cqf_threshold)
    snr_filtered = filter_by_snr(cqf_filtered, snr_threshold)
    
    # Display the filtering progression with current counts
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(f"≥ {targets_detected} detections", len(detection_filtered))
    with col2:
        st.metric(f"CQF ≥ {cqf_threshold}%", len(cqf_filtered))
    with col3:
        st.metric(f"SNR ≥ {snr_threshold}dB", len(snr_filtered))
    
    # Reset selected timestamp if it's no longer in filtered cycles
    if snr_filtered:
        current_timestamp = st.session_state.get('selected_timestamp')
        valid_timestamps = [cycle.timestamp for cycle in snr_filtered]
        
        if current_timestamp not in valid_timestamps:
            # Reset to first cycle if current selection is not valid
            st.session_state['selected_timestamp'] = snr_filtered[0].timestamp
    
    display_filtered_cycles_table(snr_filtered)
    if snr_filtered and 'selected_timestamp' in st.session_state:
        display_target_chart(snr_filtered, st.session_state['selected_timestamp'])


# Example usage - you can call this in your main app
if __name__ == "__main__":
    st.set_page_config(page_title="Baseline cycle finder", layout="wide")
    st.header("Baseline cycle finder")
    
    # Retrieve settings
    settings = Settings()
    
    # Create database engines and store in session state
    if 'platform_engine' not in st.session_state:
        st.session_state['platform_engine'] = create_engine(settings.neon_platform_connection_string)
    if 'pipeline_engine' not in st.session_state:
        st.session_state['pipeline_engine'] = create_engine(settings.neon_data_pipeline_connection_string)
 
    create_sidebar()
    
    # Display loaded data info if available
    if 'stream_id' in st.session_state:
        st.info(f"Data loaded for Stream ID: {st.session_state['stream_id']} on {st.session_state['selected_date']}")
        
        # Create filter sliders when data is loaded
        create_filter_sliders()

