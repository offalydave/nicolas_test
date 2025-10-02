from uuid import UUID
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.engine.base import Connection
from dataclasses import dataclass
import logging
from contextlib import contextmanager
from typing import Generator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Target:
    """Represents a target detection from the database."""
    timestamp: int
    name: str | None
    index: int
    x_px: float
    y_px: float
    azimuth: float
    zenith: float
    snr: float
    cqf: float
    
    def make_target_name(self) -> str:
        """Get display name for target (name if available, otherwise T_index format)."""
        return self.name or f"T_{self.index}"


@dataclass
class ProjectStreamInfo:
    """Information about a project stream."""
    project_name: str
    device_name: str
    stream_id: UUID

@dataclass
class BaselineData:
    """Baseline data container."""
    capture_time: int
    targets: list[Target]


@dataclass
class Cycle:
    """Measurement cycle data."""
    timestamp: int
    cycle_id: int


@contextmanager
def db_connection(engine: Engine) -> Generator[Connection, None, None]:
    """Context manager for database connections."""
    conn = engine.connect()
    try:
        yield conn
    finally:
        conn.close()

def get_info_for_active_projects(engine: Engine) -> list[ProjectStreamInfo]:
    """
    Get information relative to active projects.
    
    Args:
        engine: SQLAlchemy engine for the platform database
        
    Returns:
        List of ProjectStreamInfo objects containing project information
        
    Raises:
        ValueError: If the query returns no results
    """
    query = text("""
        SELECT
            s.totalite_stream_id as stream_id,
            p.name as project_name,
            d.name as device_name
        FROM
            project_device_streams s
            JOIN platform_projects p ON p.id = s.project_id
            JOIN totalite_devices d ON d.id = s.totalite_device_id
        WHERE
            s.deleted_at IS NULL
        ORDER BY
            p.name, d.name
    """)
    
    try:
        with db_connection(engine) as conn:
            result = conn.execute(query).fetchall()
            if not result:
                raise ValueError("No active projects found")
            return [ProjectStreamInfo(
                project_name=row.project_name,
                device_name=row.device_name,
                stream_id=row.stream_id
            ) for row in result]
    except Exception as e:
        logger.error(f"Error getting active projects: {str(e)}")
        raise


def get_baseline_data(engine: Engine, stream_id: UUID) -> BaselineData:
    """
    Get baseline data for a stream.
    
    Args:
        engine: SQLAlchemy engine for the data pipeline database
        stream_id: UUID of the stream
        
    Returns:
        BaselineData object containing baseline data
        
    Raises:
        ValueError: If no baseline data is found
    """
    query = text("""
        SELECT
            s.baseline
        FROM
            streams s
        WHERE
            s.id = :stream_id
            AND s.baseline IS NOT NULL
    """)
    
    try:
        with db_connection(engine) as conn:
            result = conn.execute(query, {"stream_id": stream_id}).fetchone()
            if result is None:
                raise ValueError(f"No baseline data found for stream {stream_id}")
            baseline_dict = dict(result._mapping['baseline'])
            
            # Convert baseline target dicts to Target objects
            baseline_targets = []
            for target_dict in baseline_dict['targets']:
                baseline_targets.append(Target(
                    timestamp=baseline_dict['capture_time'],
                    name=target_dict['name'],
                    index=0,  # Baseline targets don't have index from DB
                    x_px=target_dict['x_px'],
                    y_px=target_dict['y_px'],
                    azimuth=target_dict['azimuth_angle_gon'],
                    zenith=target_dict['zenith_angle_gon'],
                    snr=target_dict['signal_to_noise_ratio_db'],
                    cqf=target_dict['correlation_quality_factor'] if 'correlation_quality_factor' in target_dict else 0.0  # Old baseline format didn't have CQF, new format does
                ))
            
            return BaselineData(
                capture_time=baseline_dict['capture_time'],
                targets=baseline_targets
            )
    except Exception as e:
        logger.error(f"Error getting baseline data for stream {stream_id}: {str(e)}")
        raise

def get_target_data(engine: Engine, stream_id: UUID, start_timestamp: int, end_timestamp: int) -> list[Target]:
    """
    Get target data for a stream within a time window.
    
    Args:
        engine: SQLAlchemy engine for the data pipeline database
        stream_id: UUID of the stream
        start_timestamp: Start of the time window
        end_timestamp: End of the time window
        
    Returns:
        List of Target objects containing target data. Returns an empty list if no targets are found.
    """
    query = text("""
        SELECT
            c.capture_time as timestamp,
            tc.matched as name,
            tc.index as index,
            tc.x_px,
            tc.y_px,
            tc.azimuth_angle_raw_gon as azimuth,
            tc.zenith_angle_raw_gon as zenith,
            tc.signal_to_noise_ratio_db as snr,
            tc.correlation_quality_factor as cqf
        FROM
            streams s
            JOIN cycles c ON c.stream_id = s.id
            JOIN target_candidates tc ON tc.cycle_id = c.id
        WHERE
            s.id = :stream_id
            AND c.capture_time >= :start_timestamp
            AND c.capture_time <= :end_timestamp
        ORDER BY
            tc.matched, c.capture_time ASC
    """)
    
    with db_connection(engine) as conn:
        params = {
            "stream_id": stream_id,
            "start_timestamp": start_timestamp,
            "end_timestamp": end_timestamp
        }
        result = conn.execute(query, params).fetchall()
        return [Target(
            timestamp=row.timestamp,
            name=row.name,
            index=row._mapping['index'],  # to prevent conflict with row.index() method
            x_px=row.x_px,
            y_px=row.y_px,
            azimuth=row.azimuth,
            zenith=row.zenith,
            snr=row.snr,
            cqf=row.cqf * 100
        ) for row in result]
