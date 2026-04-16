import numpy as np
from pathlib import Path

class ReplayBuffer():
    """Dataset class for storing batches with fixed time grid
    This class is used to store and sample trajectories using disk storage.

    The replay buffer stores a set of batches with a fixed batch size. This enables to store trajectories 
    with the same time gird in a common batch and facilitates the SDE integration in torchsde by batches.
    
    Each batch of trajectories is stored as a compressed .npz file on disk.

    Attributes:
    -----------
    _batch_size: int 
        number of trajectories per batch
    _capacity: int
        maximum number of batches that can be stored in the replay buffer
    _directory: str
        name of directory where the replay buffer is stored
    _dataset: list
        list that stores the batches

    Methods:
    --------
    batch_completed:
        checks if batch is empty such that environment can be informed to generate a new time grid
    _batch_path_generator:
        returns the batch path
    __len__:
        returns the number of batches in the dataset
    _reset_current_batch:
        resets the batch storage (makes it empty) if batch size was reached for the current batch
    add_trajectory:
        add a trajectory to the current batch
    _stack_trajectories:
        stack trajectories to one batch
    _add_batch_to_dataset:
        if batch is full (batch size reached), add batch to dataset
    _save_batch_to_storage:
        save batch to disk
    _load_dataset:
        load replay buffer from disk
    _sample_batch_by_sequence_time:
        sample a batch by a sequence time (duration in seconds)
    _sample_batch_by_sequence_length:
        sample a batch by a sequence length (number of steps)
    _sample_batch_full_trajectory:
        sample the full trajectory
    _subsample_sequence;
        subsample a sequence 
    """
    
    def __init__(self, 
        directory: str,
        batch_size: int,
        capacity: int,
    ):
        """
        Docstring for __init__

        Parameters:
        -----------
        directory: str
            name of the directory
        batch_size: 
            number of trajectories per batch
        capacity:
            maximum number of batches that can be stored in the replay buffer
        """
        
        print("Initialize Replay Buffer")
    
        self._batch_size = batch_size
        self._capacity = capacity 

        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)

        self._dataset = []

        # Reset current batch 
        self._reset_current_batch() 

    @property
    def batch_completed(self) -> bool:
        """Check if batch is empty such that environment can be informed to generate a new time grid"""
        return self._current_count_in_batch == 0

    def _batch_path_generator(self, batch_identifier: int) -> Path:
        """Determine path for batch.

        The batch path consist of the complete number of steps (all trajectory steps summed up) and 
        the identifier (batch index).  The batch is stored as a compressed .npz file on disk.
        
        Parameters: 
        -----------
        batch_identifier: int
            batch index identifier

        Returns:
            batch path 
        """
        
        # Return path for batch file. <directory>/batch_<steps>_000001.npz
        return self._directory / f"batch_{self._step_counter_in_batch}_{batch_identifier:06d}.npz"
   
    def __len__(self):
        """Returns the number of batches stored in the dataset.
        """
        return len(self._dataset)
    
    def _reset_current_batch(self):
        """Reset batch storage"""
        self._current_batch = []
        self._current_count_in_batch = 0
        self._step_counter_in_batch = 0

    def add_trajectory(self, trajectory: dict):
        """Add a trajectory to current batch.
        
        Parameters:
        -----------
        trajectory: dict
            the trajectory is a dictionary. Keys are for example "action", "reward", etc.
        """

        # Add trajectory to batch
        self._current_batch.append(trajectory)

        # Increase trajectory counter in batch
        self._current_count_in_batch +=1 

        # Add the number of steps in the trajectory to the step counter
        self._step_counter_in_batch += np.stack(trajectory["reward"]).shape[0]
        
        if self._current_count_in_batch == self._batch_size: # Check if batch is full
            self._add_batch_to_dataset() # Save current batch in dataset

    def _stack_trajectories(self, trajectories: list):
        """Stack list of trajectories into a single batch.
        
        Parameters:
        -----------
        trajectories: list
            List of trajectoriess

        Returns:
            batch as a dictionary
        """

        return {
            k: np.stack([t[k] for t in trajectories], axis=0)
            for k in trajectories[0].keys()
        }

    def _add_batch_to_dataset(self):
        """Add current batch to dataset and save to disk.
        """

        # Stack trajectories in current batch since current batch is a list of trajectories
        batch = self._stack_trajectories(self._current_batch)

        # Save batch to disk and get the batch path.
        path = self._save_batch_to_storage(batch)

        # For memory efficiency, only the batch path is stored in the dataset 
        self._dataset.append(path)

        # Check if capacity is exceeded. If yes, remove the oldest batch from dataset
        if len(self._dataset) > self._capacity: 
            self._dataset.pop(0)

        # Reset batch storage
        self._reset_current_batch()

    def _save_batch_to_storage(self, batch: dict) -> Path:
        """Save the current batch to disk.
        
        Parameters:
        -----------
        batch: dict
            current batch

        Returns:
        --------
            batch_path: str
                batch path
        """
        
        # Determine index for batch file by checking the length of the dataset.
        # Old batches are overwritten if the capacity is reached.
        batch_identifier = len(self._dataset) % self._capacity 
        batch_path = self._batch_path_generator(batch_identifier)

         # Save batch to disk
        np.savez_compressed(batch_path, **batch)

        # Return batch bath instead of batch to save memorys
        return batch_path

    def load_dataset(self):
        """Load batches from disk. It does not load the data itself, only the paths to the batch files.
        """
        file_identifier = "batch_*.npz"
        # Determine all matching files in directory
        all_matching_files_in_directory = self._directory.glob(file_identifier)
        # Sort the files by their batch identifier
        batch_paths = sorted(all_matching_files_in_directory)

        # Check if the number of paths exceeds the capacity and only store the recent paths
        if len(batch_paths) > self._capacity:
            batch_paths = batch_paths[-self._capacity:] # starting from len(batch_paths)-self.capacity up to self.capacity

        # assign batch path list to dataset
        self._dataset = batch_paths

    def _sample_batch_by_sequence_time(
        self,
        sequence_time: float,
        subsampling_enabled=False,
        subsampling_time_gap_min: float = 0.01,
        subsampling_time_gap_max: float = 0.5,
    ):
        """Sample batch of sequences by a specified sequence time.

        Parameters:
        -----------
        sequence_time: float
            sequence time in seconds
        subsampling_enabled: bool
            flag to perform subsampling on the sampled sequence or not
        subsampling_time_gap_min: float
            minimum time gap between two time points during subsampling
        subsampling_time_gap_max: float
            maximum time gap between two time points during subsampling

        Returns:
        --------
            sequences: dict
                batch of sequences as a dictionary
        """
        if len(self._dataset) == 0:
            raise ValueError("No batches available")
        if subsampling_time_gap_min > subsampling_time_gap_max:
            raise ValueError(
                "subsampling_time_gap_min must be smaller than or equal to "
                "subsampling_time_gap_max."
            )

        # Determine random batch path
        path = np.random.choice(self._dataset) 

        # Load a dict like object
        data = np.load(path) 
        # Convert to a regular dict
        batch = {k: data[k] for k in data.files} 

        B, T = batch["time"].shape
        times = batch["time"][0] # all elements in the batch have the same time grid

        # Determine minimum time and maximum time for the starting time 
        t_minimum = times[0]
        t_maximum = times[-1] - sequence_time

        # Check if sequence time is longer than the episode. 
        # If yes, starting time is t_min and therefore starting inde is 0 and ending index T.
        if t_maximum <= t_minimum:
            start_index = 0
            end_index = T
        else:
            # Sample starting time uniformly 
            t0 = np.random.uniform(t_minimum, t_maximum)

            # Determine first index with time >= t0
            start_index = np.searchsorted(times, t0, side="left")

            t_end = t0 + sequence_time
            end_index = np.searchsorted(times, t_end, side="right")

        # Sample all sequences starting from index i0 to i1
        sequences= {
            key: value[:, start_index:end_index, ...]
            for key, value in batch.items()
        }

        # Perform subsampling on sequences if enabled 
        if subsampling_enabled:
            sequences = self._subsample_sequence(
                sequences,
                time_gap_min=subsampling_time_gap_min,
                time_gap_max=subsampling_time_gap_max,
            )
        
        return sequences
    
    def sample_batch_by_sequence_length(self, sequence_length: int):
        """Sample batch of sequences by a specified sequence length (number of steps). 

        Subsampling logic is not added to this sampling method since subsampling is only interesting
        in an irregular setting.
        
        Parameters:
        -----------
        sequence_length: int
            sequence length as number of steps
        Returns:
        --------
            sequences: dict
                batch of sequences as a dictionary
        """
        if len(self._dataset) == 0:
            raise ValueError("No batches available")

        # Determine random batch path
        path = np.random.choice(self._dataset)

        # Load  a dict like object
        data = np.load(path)  
        # Convert to a regular dict
        batch = {key: data[key] for key in data.files} 

        # Get batch size and trajectory length
        B, T = batch["obs"].shape[:2] 

        # Check if sequence length is valid. If not choose T as sequence length.
        if sequence_length > T: 
            sequence_length = T

        # Determine random starting index for sequence
        start_index = np.random.randint(0, T - sequence_length + 1) 

        # Sample all sequences starting from index start with length=sequence length 
        sequences =  {
            key: value[:, start_index:start_index + sequence_length, ...]
            for key, value in batch.items()
        }

        return sequences
    
    def sample_batch_full_trajectory(self):
        """Sample full trajectory.
        """
        if len(self._dataset) == 0:
            raise ValueError("Replay buffer is empty")

        # Determine random batch path
        path = np.random.choice(self._dataset) 

        # Load a dict like object
        data = np.load(path)  
        # Convert to a regular dict
        batch = {key: data[key] for key in data.files}

        return batch
    
    def _subsample_sequence(self, batch:dict, time_gap_min: float =0.01, time_gap_max:float=0.5):
        """Subsample a sequence 

        A minimum and maximum time gap is defined.
        With those values, we start with the first time point in the sequence and sample
        a random time gap. Until the time gap is reached, time steps in between are skipped.
        Example: 
        - t=[0,0.2,0.5,0.6,0.7]
        - Current time point is 0.2
        - Time gap is 0.35 seconds => the data at time point 0.5 is skipped and the next time point
          is 0.6
        
        Parameters:
        -----------
        batch: dict
            batch of sequences
        time_gap_min: float
            minimum time gap between two time steps
        time_gap_max: float
            maximum time gap between two time steps

        Returns:
        --------
        subsampled_sequences: dict
            subsampled sequences

        """
        times = batch["time"]  
   
        # Get time grid
        time_grid = times[0] # all elements in the batch have the same time grid   
        T = time_grid.shape[0]

        # Store all indices of times steps that remain after the subsampling
        indices_to_keep = [0]
        
        # Initialize last time step
        last_time_step = time_grid[0]

        # Determine next time step
        next_time_step = last_time_step + np.random.uniform(time_gap_min, time_gap_max)

        for i in range(1, T):
            # Check if time gap is reached. If not, skip time step.
            if time_grid[i] >= next_time_step:
                indices_to_keep.append(i)
                last_time_step = time_grid[i]
                # Determine next time step
                next_time_step = last_time_step + np.random.uniform(time_gap_min, time_gap_max)

        # Convert list to numpy array
        indices_to_keep = np.array(indices_to_keep, dtype=np.int64)

        # Sample subsampled sequence using the indices to keep
        subsampled_sequences = {
            key: value[:, indices_to_keep, ...]
            for key, value in batch.items()
        }
        return subsampled_sequences