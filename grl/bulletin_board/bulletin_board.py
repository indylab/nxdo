from threading import RLock
import numpy as np

"""
Currently not used by anything. Wrote it and then decided I didn't need it.
"""


class BulletinBoard(object):
    """
    BulletinBoard an algorithm agnostic in-memory key-value store designed for
    facilitating synchronization and status updates between remote workers.
    All operations are atomic.
    """

    def __init__(self):
        self._modification_lock = RLock()
        self._board = {}

    def Set(self, key, value, overwrite=True):
        with self._modification_lock:
            if not overwrite and key in self._board:
                return False
            if value is None:
                del self._board[key]
            else:
                self._board[key] = value
            return True

    def Get(self, key):
        with self._modification_lock:
            return self._board.get(key, default=None)

    def SetMultiple(self, set_dict, overwrite=True, do_nothing_if_any_previously_set=False):
        with self._modification_lock:
            if do_nothing_if_any_previously_set:
                for key in set_dict.keys():
                    if key in self._board:
                        return {key: False for key in set_dict.keys()}
            result_dict = {}
            for key, val in set_dict.items():
                result_dict[key] = self.Set(key=key, value=val, overwrite=overwrite)
            return result_dict

    def GetMultiple(self, key_list):
        with self._modification_lock:
            result_dict = {}
            for key in key_list:
                result_dict[key] = self.Get(key=key)
            return result_dict

    def GetAndSet(self, key, value):
        with self._modification_lock:
            if key in self._board:
                orig_val = self._board[key]
            else:
                orig_val = None
            self._board[key] = value
            return orig_val

    def GetAndIncrementInt(self, key, incr_amount=1, default_if_not_set=0):
        with self._modification_lock:
            orig_val = self.Get(key=key)
            if orig_val is not None and not np.issubdtype(type(orig_val), np.integer):
                raise TypeError(f"Value for key {key} is needs to be an integer. "
                                f"It's actually of type {type(orig_val)}.")
            if orig_val is None:
                self.Set(key=key, value=int(default_if_not_set), overwrite=True)
            else:
                new_val = int(orig_val + incr_amount)
                self.Set(key=key, value=new_val, overwrite=True)
            return orig_val

    def CompareAndSetMultiple(self, compare_dict, set_dict,
                              overwrite=True, do_nothing_if_any_previously_set=False):
        with self._modification_lock:
            orig_val_dict = {}
            all_equal = True
            for key, compare_val in compare_dict.items():
                current_val_for_key = self.Get(key=key)
                if current_val_for_key != compare_val:
                    all_equal = False
                orig_val_dict[key] = current_val_for_key
            if all_equal:
                set_result_dict = self.SetMultiple(set_dict=set_dict, overwrite=overwrite,
                                              do_nothing_if_any_previously_set=do_nothing_if_any_previously_set)
            else:
                set_result_dict = {set_key: False for set_key in set_dict.keys()}
            return orig_val_dict, set_result_dict
