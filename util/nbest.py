import numpy as np


class NBest:
    def __init__(self, N=3, less_is_better=True):
        self.N = N
        self.LESS_IS_BETTER = less_is_better

    def set_N(self, N):
        self.N = N

    def set_LESS_IS_BETTER(self, better=True):
        self.LESS_IS_BETTER = better

    def get_N(self):
        return self.N

    def get_LESS_IS_BETTER(self):
        return self.LESS_IS_BETTER

    def get_best(self, measure_list):
        try:
            assert (len(measure_list) is not 0), 'No value in measure_list!'
        except AssertionError as error:
            print(error)
        else:
            if self.LESS_IS_BETTER:
                best = min(measure_list)
            else:
                best = max(measure_list)
            return best

    def get_worst(self, measure_list):
        try:
            assert (len(measure_list) is not 0), 'No value in measure_list!'
        except AssertionError as error:
            print(error)
        else:
            if self.LESS_IS_BETTER:
                worst = max(measure_list)
            else:
                worst = min(measure_list)
            worst_ind = measure_list.index(worst)
            return worst, worst_ind

    def save_model(self, measure_list, new_measure=0):
        if len(measure_list) == 0:
            return True
        else:
            best = self.get_best(measure_list)
            if self.LESS_IS_BETTER:
                if new_measure < best:
                    return True
                else:
                    return False
            else:
                if new_measure > best:
                    return True
                else:
                    return False

    def pop_worst(self, measure_list, epoch_list, wb_path_list):
        delete_list = []
        while len(measure_list) > self.N:
            worst, worst_ind = self.get_worst(measure_list)
            measure_list.pop(worst_ind)
            epoch_list.pop(worst_ind)
            delete_wb = wb_path_list.pop(worst_ind)
            delete_list.append(delete_wb)

        return measure_list, epoch_list, wb_path_list, delete_list

    def sort_measure(self, measure_list, epoch_list, wb_list):
        # Worst to best
        try:
            assert (len(measure_list) is not 0), 'measure list empty, nothing to sort!'
        except AssertionError as error:
            print(error)
        else:
            if len(measure_list) > 1:
                sorted_index = np.argsort(np.array(measure_list))
                sorted_index.tolist()
                unsorted_measures = measure_list.copy()
                unsorted_epochs = epoch_list.copy()
                unsorted_wb_list = wb_list.copy()

                i = 0
                for ind in sorted_index:
                    measure_list[i] = unsorted_measures[ind]
                    wb_list[i] = unsorted_wb_list[ind]
                    epoch_list[i] = unsorted_epochs[ind]
                    i += 1

                if self.LESS_IS_BETTER:
                    # flip the list to descending order.
                    measure_list = measure_list[::-1]
                    wb_list = wb_list[::-1]
                    epoch_list = epoch_list[::-1]

            return measure_list, epoch_list, wb_list
