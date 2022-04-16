import os
import pickle
import warnings

class MemoryDumper:
    """
    Utility to dump memory records into a file.
    """

    def __init__(self, file_name_base, header={}, file_bytes_goal=100*1024*1024, file_limit=10, save_file_func=None):
        self.file_name_base = file_name_base
        self.file_bytes_goal = file_bytes_goal
        self.header = header
        self.file_limit = file_limit
        self.save_file_func = save_file_func

        self.file = None
        self.next_file_idx = 0

    def dump(self, record):
        # if the current batch is full, close it
        if self.file is not None and self.file.tell() > self.file_bytes_goal:
            self.close()

        # if there is no open batch, create a new one
        if self.file is None:
            if self.next_file_idx >= self.file_limit:
                warnings.warn(f'MemoryDumper reached the limit of {self.file_limit} files. Memory dumping stopped.')
                return

            self.cur_file_name = "{}.{}".format(self.file_name_base, self.next_file_idx)
            self.file = open(self.cur_file_name, 'wb')
            self.next_file_idx = self.next_file_idx + 1
            pickle.dump(self.header, self.file)

        # pickle the record into the current batch
        pickle.dump(record, self.file)

    def sync(self):
        if self.file is not None:
            # close the file
            self.file.close()
            self.file = None

            # compress and upload the file
            if self.save_file_func is not None:
                self.save_file_func(self.cur_file_name)
                if self.cur_file_name and os.path.exists(self.cur_file_name):
                    os.remove(self.cur_file_name)

            # mark the batch as finished
            self.cur_file_name = None
    
    def close(self):
        self.sync()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
