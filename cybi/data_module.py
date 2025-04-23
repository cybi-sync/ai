import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA, FastICA, KernelPCA
from sklearn.manifold import TSNE, Isomap
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_regression
from sklearn.impute import KNNImputer
import joblib
import os
from datetime import datetime
import logging
from tqdm import tqdm
from pyod.models.knn import KNN as PyodKNN
from pyod.models.iforest import IForest
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.piecewise import PiecewiseAggregateApproximation
from tslearn.piecewise import SymbolicAggregateApproximation
import pywt

logger = logging.getLogger(__name__)

class CYBIDataProcessor:
    """
    Advanced data processor for CYBI smartshoe data, implementing multiple
    processing pipelines for extracting complex features from various sensors.
    """
    
    def __init__(self, config=None):
        self.config = config or {
            'scaling_method': 'robust',  # Options: 'standard', 'minmax', 'robust'
            'dimension_reduction': 'pca',  # Options: 'pca', 'kpca', 'ica', 'tsne', 'isomap'
            'imputation_method': 'knn',  # Options: 'knn', 'mean', 'median', 'interpolate'
            'anomaly_detection': 'iforest',  # Options: 'knn', 'iforest'
            'feature_selection_method': 'mutual_info',  # Options: 'f_classif', 'mutual_info'
            'n_components': 0.95,  # Proportion of variance to retain or number of components
            'wavelet_family': 'db4',  # Wavelet family for wavelet transform
            'paa_segments': 10,  # Number of segments for PAA
            'sax_alphabet_size': 8,  # Alphabet size for SAX
            'time_features': True,  # Extract time-based features
            'frequency_features': True,  # Extract frequency-domain features
            'statistical_features': True,  # Extract statistical features
            'sensor_fusion': True,  # Apply sensor fusion techniques
            'calibration': True,  # Apply sensor calibration
            'motion_segmentation': True,  # Segment motion patterns
            'gait_analysis': True,  # Extract gait analysis features
            'pressure_mapping': True,  # Create pressure maps
            'stance_analysis': True,  # Analyze stance phases
            'custom_filters': [
                {'type': 'butterworth', 'cutoff': 5.0, 'order': 4},
                {'type': 'savgol', 'window_size': 15, 'poly_order': 3}
            ],
            'save_intermediates': False,  # Save intermediate processing results
            'cache_dir': './cache/data_processing'
        }
        
        self.scalers = {}
        self.dim_reducers = {}
        self.imputers = {}
        self.anomaly_detectors = {}
        self.feature_selectors = {}
        
        # Create cache directory if needed
        if self.config['save_intermediates']:
            os.makedirs(self.config['cache_dir'], exist_ok=True)
        
        # Initialize processors
        self._init_processors()
    
    def _init_processors(self):
        """Initialize all data processing components."""
        # Initialize scalers
        scaling_method = self.config['scaling_method']
        if scaling_method == 'standard':
            self.scalers['main'] = StandardScaler()
        elif scaling_method == 'minmax':
            self.scalers['main'] = MinMaxScaler()
        elif scaling_method == 'robust':
            self.scalers['main'] = RobustScaler()
        
        # Special scaler for time series
        self.scalers['time_series'] = TimeSeriesScalerMeanVariance()
        
        # Initialize dimension reducers
        dim_red = self.config['dimension_reduction']
        n_components = self.config['n_components']
        if dim_red == 'pca':
            self.dim_reducers['main'] = PCA(n_components=n_components)
        elif dim_red == 'kpca':
            self.dim_reducers['main'] = KernelPCA(n_components=n_components, kernel='rbf')
        elif dim_red == 'ica':
            self.dim_reducers['main'] = FastICA(n_components=n_components)
        elif dim_red == 'tsne':
            self.dim_reducers['main'] = TSNE(n_components=min(3, n_components))
        elif dim_red == 'isomap':
            self.dim_reducers['main'] = Isomap(n_components=n_components)
        
        # Initialize imputers
        if self.config['imputation_method'] == 'knn':
            self.imputers['main'] = KNNImputer(n_neighbors=5)
        
        # Initialize anomaly detectors
        if self.config['anomaly_detection'] == 'knn':
            self.anomaly_detectors['main'] = PyodKNN(contamination=0.05)
        elif self.config['anomaly_detection'] == 'iforest':
            self.anomaly_detectors['main'] = IForest(contamination=0.05, random_state=42)
        
        # Initialize feature selectors
        if self.config['feature_selection_method'] == 'f_classif':
            self.feature_selectors['weight'] = SelectKBest(f_classif, k=20)
            self.feature_selectors['health'] = SelectKBest(f_classif, k=30)
        elif self.config['feature_selection_method'] == 'mutual_info':
            self.feature_selectors['weight'] = SelectKBest(mutual_info_regression, k=20)
            self.feature_selectors['health'] = SelectKBest(mutual_info_regression, k=30)
        
        # Initialize time series transformation
        self.paa = PiecewiseAggregateApproximation(n_segments=self.config['paa_segments'])
        self.sax = SymbolicAggregateApproximation(n_segments=self.config['paa_segments'], 
                                                 alphabet_size_avg=self.config['sax_alphabet_size'])
    
    def _apply_filters(self, data):
        """Apply multiple signal filters to the data."""
        filtered_data = data.copy()
        
        for filter_config in self.config['custom_filters']:
            if filter_config['type'] == 'butterworth':
                b, a = signal.butter(filter_config['order'], 
                                    filter_config['cutoff'] / (0.5 * 100),  # Assuming 100Hz sample rate
                                    btype='low')
                for col in filtered_data.columns:
                    filtered_data[col] = signal.filtfilt(b, a, filtered_data[col])
            
            elif filter_config['type'] == 'savgol':
                for col in filtered_data.columns:
                    filtered_data[col] = signal.savgol_filter(filtered_data[col], 
                                                           filter_config['window_size'],
                                                           filter_config['poly_order'])
            
            elif filter_config['type'] == 'median':
                for col in filtered_data.columns:
                    filtered_data[col] = signal.medfilt(filtered_data[col], 5)
                    
        return filtered_data
    
    def _extract_time_features(self, data):
        """Extract complex time-domain features from sensor data."""
        features = pd.DataFrame(index=data.index[::100])  # Downsample for feature extraction
        window_size = 100
        
        # Process data in windows
        for i in tqdm(range(0, len(data) - window_size, window_size), desc="Extracting time features"):
            window = data.iloc[i:i+window_size]
            window_features = {}
            
            for col in window.columns:
                series = window[col].values
                
                # Basic statistics
                window_features[f"{col}_mean"] = np.mean(series)
                window_features[f"{col}_std"] = np.std(series)
                window_features[f"{col}_max"] = np.max(series)
                window_features[f"{col}_min"] = np.min(series)
                window_features[f"{col}_range"] = np.ptp(series)
                window_features[f"{col}_median"] = np.median(series)
                window_features[f"{col}_mad"] = np.median(np.abs(series - np.median(series)))
                window_features[f"{col}_iqr"] = np.percentile(series, 75) - np.percentile(series, 25)
                
                # Advanced statistics
                window_features[f"{col}_kurtosis"] = pd.Series(series).kurtosis()
                window_features[f"{col}_skew"] = pd.Series(series).skew()
                window_features[f"{col}_energy"] = np.sum(series**2) / len(series)
                window_features[f"{col}_peak"] = np.max(np.abs(series))
                window_features[f"{col}_crest"] = np.max(np.abs(series)) / np.sqrt(np.mean(series**2))
                
                # Time series complexity
                window_features[f"{col}_crossing_rate"] = np.sum(np.diff(np.signbit(series).astype(int))) / (len(series) - 1)
                
                # Autocorrelation at lag 1
                window_features[f"{col}_autocorr"] = np.corrcoef(series[:-1], series[1:])[0, 1] if len(series) > 1 else 0
                
                # Wavelet decomposition - extract energy from wavelet coefficients
                if len(series) >= 4:  # Minimum length required for wavelet transform
                    coeffs = pywt.wavedec(series, self.config['wavelet_family'], level=3)
                    for j, coeff in enumerate(coeffs):
                        window_features[f"{col}_wavelet_energy_{j}"] = np.sum(coeff**2) / len(coeff) if len(coeff) > 0 else 0
            
            # Append to features DataFrame
            if i // window_size < len(features):
                features.iloc[i // window_size] = pd.Series(window_features)
        
        return features
    
    def _extract_frequency_features(self, data):
        """Extract complex frequency-domain features using Fourier and wavelet transforms."""
        features = pd.DataFrame(index=data.index[::100])  # Downsample
        window_size = 100
        
        for i in tqdm(range(0, len(data) - window_size, window_size), desc="Extracting frequency features"):
            window = data.iloc[i:i+window_size]
            window_features = {}
            
            for col in window.columns:
                series = window[col].values
                
                # Fourier transform
                if len(series) > 0:
                    fft_vals = np.abs(np.fft.rfft(series))
                    fft_freqs = np.fft.rfftfreq(len(series), d=0.01)  # Assuming 100Hz
                    
                    # Extract frequency domain features
                    window_features[f"{col}_dom_freq"] = fft_freqs[np.argmax(fft_vals)] if len(fft_vals) > 0 else 0
                    window_features[f"{col}_freq_mean"] = np.mean(fft_vals) if len(fft_vals) > 0 else 0
                    window_features[f"{col}_freq_std"] = np.std(fft_vals) if len(fft_vals) > 0 else 0
                    window_features[f"{col}_freq_skew"] = pd.Series(fft_vals).skew() if len(fft_vals) > 0 else 0
                    window_features[f"{col}_freq_kurt"] = pd.Series(fft_vals).kurtosis() if len(fft_vals) > 0 else 0
                    
                    # Spectral bands energy (assuming human movement relevant bands)
                    if len(fft_freqs) > 1:
                        bounds = [0, 0.5, 1.5, 5, 10, 20]  # Movement frequency bands in Hz
                        for j in range(len(bounds)-1):
                            band_indices = np.logical_and(fft_freqs >= bounds[j], fft_freqs < bounds[j+1])
                            band_energy = np.sum(fft_vals[band_indices]**2) / len(fft_vals) if np.any(band_indices) else 0
                            window_features[f"{col}_band_{bounds[j]}-{bounds[j+1]}_energy"] = band_energy
                    
                    # Spectral entropy
                    if np.sum(fft_vals) > 0:
                        norm_fft = fft_vals / np.sum(fft_vals)
                        spectral_entropy = -np.sum(norm_fft * np.log2(norm_fft + 1e-10))
                        window_features[f"{col}_spectral_entropy"] = spectral_entropy
            
            # Append to features DataFrame
            if i // window_size < len(features):
                features.iloc[i // window_size] = pd.Series(window_features)
        
        return features
    
    def _extract_gait_features(self, data):
        """Extract specialized gait analysis features from pressure and motion sensors."""
        # This would be a complex implementation based on biomechanical principles
        features = pd.DataFrame(index=data.index[::100])  # Downsample
        
        # Simplified implementation - in a real system this would be much more complex
        # and based on specialized gait analysis algorithms
        pressure_cols = [col for col in data.columns if 'pressure' in col.lower()]
        accel_cols = [col for col in data.columns if 'accel' in col.lower()]
        gyro_cols = [col for col in data.columns if 'gyro' in col.lower()]
        
        window_size = 200  # Larger window for gait cycle detection
        
        if not pressure_cols or not accel_cols:
            logger.warning("Missing pressure or acceleration data for gait analysis")
            return pd.DataFrame()
        
        for i in tqdm(range(0, len(data) - window_size, window_size), desc="Extracting gait features"):
            window = data.iloc[i:i+window_size]
            window_features = {}
            
            # Calculate pressure distribution features
            if pressure_cols:
                pressure_data = window[pressure_cols]
                window_features["pressure_asymmetry"] = self._calculate_pressure_asymmetry(pressure_data)
                window_features["pressure_variability"] = np.mean([np.std(window[col]) for col in pressure_cols])
                window_features["pressure_max_ratio"] = self._calculate_pressure_ratio(pressure_data)
            
            # Calculate step detection from accelerometer data
            if accel_cols and len(accel_cols) >= 3:
                # Assuming XYZ order for accelerometer columns
                accel_data = window[accel_cols].values
                steps, step_times = self._detect_steps(accel_data)
                window_features["step_count"] = steps
                window_features["step_frequency"] = steps / (window_size/100) if window_size > 0 else 0  # Assuming 100Hz
                window_features["step_regularity"] = np.std(step_times) if step_times else 0
                
                # Calculate impact forces
                if steps > 0:
                    impacts = self._calculate_impacts(accel_data)
                    window_features["impact_mean"] = np.mean(impacts) if impacts else 0
                    window_features["impact_max"] = np.max(impacts) if impacts else 0
            
            # Calculate rotation features from gyroscope data
            if gyro_cols and len(gyro_cols) >= 3:
                gyro_data = window[gyro_cols].values
                window_features["rotation_energy"] = np.sum(gyro_data**2) / len(gyro_data)
                window_features["rotation_symmetry"] = self._calculate_rotation_symmetry(gyro_data)
            
            # Append to features DataFrame
            if i // window_size < len(features):
                features.iloc[i // window_size] = pd.Series(window_features)
        
        return features
    
    def _calculate_pressure_asymmetry(self, pressure_data):
        """Calculate asymmetry between left and right foot pressure sensors."""
        # Simplified implementation - would be more complex in reality
        left_cols = [col for col in pressure_data.columns if 'left' in col.lower()]
        right_cols = [col for col in pressure_data.columns if 'right' in col.lower()]
        
        if not left_cols or not right_cols:
            return 0
        
        left_pressure = np.mean([pressure_data[col].mean() for col in left_cols])
        right_pressure = np.mean([pressure_data[col].mean() for col in right_cols])
        
        if left_pressure + right_pressure == 0:
            return 0
            
        return abs(left_pressure - right_pressure) / (left_pressure + right_pressure)
    
    def _calculate_pressure_ratio(self, pressure_data):
        """Calculate ratio of maximum to average pressure."""
        if pressure_data.empty:
            return 0
            
        max_pressure = pressure_data.max().max()
        mean_pressure = pressure_data.mean().mean()
        
        if mean_pressure == 0:
            return 0
            
        return max_pressure / mean_pressure
    
    def _detect_steps(self, accel_data):
        """Detect steps from accelerometer data using peak detection."""
        if len(accel_data) < 10:
            return 0, []
            
        # Calculate magnitude of acceleration
        accel_mag = np.sqrt(np.sum(accel_data**2, axis=1))
        
        # Apply smoothing
        accel_mag_smooth = signal.savgol_filter(accel_mag, 11, 2)
        
        # Find peaks with minimum height and distance
        peaks, _ = signal.find_peaks(accel_mag_smooth, height=1.1*np.mean(accel_mag_smooth), 
                                    distance=20)  # Minimum 0.2s between steps at 100Hz
        
        # Calculate time between steps
        step_times = np.diff(peaks) / 100  # Assuming 100Hz sampling
        
        return len(peaks), step_times.tolist()
    
    def _calculate_impacts(self, accel_data):
        """Calculate impact forces from acceleration data."""
        accel_mag = np.sqrt(np.sum(accel_data**2, axis=1))
        
        # Find peaks representing foot impacts
        impacts, _ = signal.find_peaks(accel_mag, height=1.5*np.mean(accel_mag), distance=20)
        
        # Get the magnitudes of the impacts
        impact_forces = accel_mag[impacts] if len(impacts) > 0 else []
        
        return impact_forces
    
    def _calculate_rotation_symmetry(self, gyro_data):
        """Calculate symmetry in rotational movements."""
        if len(gyro_data) < 20:
            return 0
            
        # Take the first half and second half of the window to compare symmetry
        half_point = len(gyro_data) // 2
        first_half = gyro_data[:half_point]
        second_half = gyro_data[half_point:2*half_point]
        
        if len(first_half) != len(second_half):
            second_half = second_half[:len(first_half)]
        
        # Compare the energy in each axis
        if len(first_half) == 0 or len(second_half) == 0:
            return 0
            
        energy_first = np.sum(first_half**2, axis=0)
        energy_second = np.sum(second_half**2, axis=0)
        
        # Calculate symmetry score (1 = perfect symmetry, 0 = complete asymmetry)
        symmetry = 1 - np.mean(np.abs(energy_first - energy_second) / (energy_first + energy_second + 1e-10))
        
        return symmetry
    
    def process(self, raw_data):
        """Main processing pipeline that applies all configured processing steps."""
        # Input validation
        if raw_data is None or len(raw_data) == 0:
            raise ValueError("Empty or None input data")
        
        logger.info(f"Processing {len(raw_data)} samples of CYBI smartshoe data")
        
        # Step 1: Apply calibration if enabled
        if self.config['calibration']:
            data = self._apply_calibration(raw_data)
        else:
            data = raw_data.copy()
        
        # Step 2: Apply filters to clean the data
        data = self._apply_filters(data)
        
        # Step 3: Handle missing values
        if self.config['imputation_method'] == 'knn':
            data = pd.DataFrame(
                self.imputers['main'].fit_transform(data),
                columns=data.columns,
                index=data.index
            )
        
        # Step 4: Detect and remove anomalies
        if self.config['anomaly_detection'] in self.anomaly_detectors:
            detector = self.anomaly_detectors[self.config['anomaly_detection']]
            anomaly_scores = detector.fit_predict(data)
            data['anomaly_score'] = anomaly_scores
            # Flag samples with high anomaly scores
            data['is_anomaly'] = anomaly_scores > 1.5
            
            # For extreme anomalies, we might want to exclude them
            extreme_anomalies = data['anomaly_score'] > 2.5
            if sum(extreme_anomalies) > 0:
                logger.warning(f"Detected {sum(extreme_anomalies)} extreme anomalies in the data")
            
            # Keep anomaly score but remove flag column
            data = data.drop(columns=['is_anomaly'])
        
        # Step 5: Extract features from different domains
        features = {}
        
        if self.config['time_features']:
            features['time'] = self._extract_time_features(data)
        
        if self.config['frequency_features']:
            features['frequency'] = self._extract_frequency_features(data)
        
        if self.config['gait_analysis']:
            features['gait'] = self._extract_gait_features(data)
        
        # Step 6: Combine all features
        combined_features = pd.concat([df for df in features.values()], axis=1)
        
        # Step 7: Apply feature scaling
        scaled_features = pd.DataFrame(
            self.scalers['main'].fit_transform(combined_features.fillna(0)),
            columns=combined_features.columns,
            index=combined_features.index
        )
        
        # Step 8: Apply dimension reduction if needed
        if self.config['dimension_reduction'] in self.dim_reducers:
            reducer = self.dim_reducers[self.config['dimension_reduction']]
            reduced_data = reducer.fit_transform(scaled_features.fillna(0))
            
            # Create DataFrame with reduced dimensions
            column_names = [f"component_{i+1}" for i in range(reduced_data.shape[1])]
            reduced_features = pd.DataFrame(
                reduced_data,
                columns=column_names,
                index=scaled_features.index
            )
            
            # Keep track of explained variance if using PCA
            if self.config['dimension_reduction'] == 'pca':
                explained_variance = reducer.explained_variance_ratio_
                logger.info(f"PCA explained variance: {np.sum(explained_variance):.4f} with {len(explained_variance)} components")
            
            # Combine reduced features with original anomaly scores if available
            if 'anomaly_score' in data.columns:
                reduced_features['anomaly_score'] = data.loc[reduced_features.index, 'anomaly_score']
            
            processed_data = reduced_features
        else:
            processed_data = scaled_features
        
        # Step 9: Save intermediate results if configured
        if self.config['save_intermediates']:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            joblib.dump(self.scalers, f"{self.config['cache_dir']}/scalers_{timestamp}.pkl")
            joblib.dump(self.dim_reducers, f"{self.config['cache_dir']}/dim_reducers_{timestamp}.pkl")
            processed_data.to_csv(f"{self.config['cache_dir']}/processed_data_{timestamp}.csv")
        
        logger.info(f"Data processing complete. Generated {processed_data.shape[1]} features from {len(raw_data)} samples")
        
        return processed_data
    
    def _apply_calibration(self, data):
        """Apply sensor calibration based on calibration parameters."""
        # In a real system, this would apply sensor-specific calibration
        # Here we'll implement a simplified version for illustration
        calibrated_data = data.copy()
        
        # Apply calibration offsets and scaling factors
        # (These would typically be determined through a calibration procedure)
        for col in calibrated_data.columns:
            if 'accel' in col.lower() or 'gyro' in col.lower():
                # Apply typical IMU calibration - remove offset and apply scaling
                offset = np.mean(calibrated_data[col].head(100))  # Use first 100 samples for offset
                calibrated_data[col] = (calibrated_data[col] - offset) * 1.02  # 2% scaling adjustment
            
            elif 'pressure' in col.lower():
                # Apply pressure sensor calibration
                # Here we're assuming a simple linear calibration
                calibrated_data[col] = calibrated_data[col] * 1.05 - 5  # Example adjustment
        
        return calibrated_data

    def save(self, filepath):
        """Save the processor state and configuration."""
        joblib.dump({
            'config': self.config,
            'scalers': self.scalers,
            'dim_reducers': self.dim_reducers,
            'imputers': self.imputers,
            'anomaly_detectors': self.anomaly_detectors,
            'feature_selectors': self.feature_selectors
        }, filepath)
    
    @classmethod
    def load(cls, filepath):
        """Load a processor from a saved state."""
        saved_state = joblib.load(filepath)
        processor = cls(config=saved_state['config'])
        processor.scalers = saved_state['scalers']
        processor.dim_reducers = saved_state['dim_reducers']
        processor.imputers = saved_state['imputers']
        processor.anomaly_detectors = saved_state['anomaly_detectors']
        processor.feature_selectors = saved_state['feature_selectors']
        return processor 