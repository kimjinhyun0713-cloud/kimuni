import pickle


class PKL:
    @staticmethod
    def _load(pkl):
        with open(pkl, "rb") as f:
            return pickle.load(f)

    def __init__(self, pkl):
        self.stats = PKL._load(pkl)


class Ab_stats(PKL):
    total_ca_label = {}
    total_co3_label = {}
    total_all_label = {}
    ab_ca_type = {}
    ab_co3_type = {}
    all_type = {}

    @property
    def co3_label(self):
        return self._co3_label

    @co3_label.setter
    def co3_label(self, val):
        self._co3_label[val] = self._co3_label.get(val, 0) + 1

    @property
    def all_label(self):
        return self._all_label

    @all_label.setter
    def all_label(self, val):
        self._all_label[val] = self._all_label.get(val, 0) + 1

    @property
    def ca_label(self):
        return self._ca_label

    @ca_label.setter
    def ca_label(self, val):
        self._ca_label[val] = self._ca_label.get(val, 0) + 1

    def __init__(self, pkl):
        super().__init__(pkl)
        self._setting_()
        self.analyze()
        self.merge2class()

    def _setting_(self):
        if isinstance(self.stats[0][0], list):
            self.connected_stats = self.stats[
                0
            ]  # (要修正)csh_analyzer.stdout_statistics
            self.all_stats = self.stats[1]  #  (要修正)csh_analyzer.stdout_statistics
        else:
            self.connected_stats = self.stats[
                1
            ]  # (要修正)csh_analyzer.stdout_statistics
            self.all_stats = self.stats[0]  #  (要修正)csh_analyzer.stdout_statistics
        self.co3_stats = [stats[0] for stats in self.connected_stats]
        self.ca_stats = [stats[1:] for stats in self.connected_stats]
        self._co3_label = {}
        self._ca_label = {}
        self._all_label = {}

    def analyze(self):
        for stat in self.ca_stats:
            for stat_ in stat:
                key = tuple(sorted(stat_.items()))
                self.ca_label = key

        for stat_ in self.all_stats:
            key = tuple(sorted(stat_.items()))
            self.all_label = key

        for stat in self.co3_stats:
            key = tuple(sorted(stat.items()))
            self.co3_label = key

    def merge2class(self):
        for k, v in self.co3_label.items():
            Ab_stats.total_co3_label[k] = Ab_stats.total_co3_label.get(k, 0) + v
        for k, v in self.ca_label.items():
            Ab_stats.total_ca_label[k] = Ab_stats.total_ca_label.get(k, 0) + v
        for k, v in self.all_label.items():
            Ab_stats.total_all_label[k] = Ab_stats.total_all_label.get(k, 0) + v

    def set_type(self):
        for label, val in Ab_stats.total_ca_label.items():
            type_ = self.label2Catype(label)
            Ab_stats.ab_ca_type[type_] = Ab_stats.ab_ca_type.get(type_, 0) + val
        for label, val in Ab_stats.total_co3_label.items():
            type_ = self.label2CO3type(label)
            Ab_stats.ab_co3_type[type_] = Ab_stats.ab_co3_type.get(type_, 0) + val
        for label, val in Ab_stats.total_all_label.items():
            type_ = self.label2Catype(label)
            Ab_stats.all_type[type_] = Ab_stats.all_type.get(type_, 0) + val
        print(Ab_stats.ab_ca_type)
        print(Ab_stats.ab_co3_type)
        print(Ab_stats.all_type)

    def label2Catype(self, label):
        sort_key = ["CT_O", "PT_O", "NT_O", "bt_O", "BT_O"]
        order = {k: i for i, k in enumerate(sort_key)}
        label = sorted(label, key=lambda x: order.get(x[0], float("inf")))
        label = label[0]
        if label[0] == "free":
            return "Ca_free"
        elif label[0] == "CT_O":
            return "Ca_type5"
        elif label[0] == "PT_O":
            return "Ca_type4"
        elif (label[0] == "NT_O") and label[1] == 2:
            return "Ca_type3"
        elif label[0] in ("bT_O", "NT_O"):
            return "Ca_type2"
        elif label[0] == "BT_O":
            return "Ca_type1"
        else:
            raise ValueError("not label")

    def label2CO3type(self, label):
        if label == ():
            return "H_free"
        sort_key = ["PT_H", "NT_H", "NT_H_", "bt_H", "bt_H_", "BT_H", "BT_H_"]
        order = {k: i for i, k in enumerate(sort_key)}
        label = sorted(label, key=lambda x: order.get(x[0], float("inf")))
        label = label[0]
        if label[0] == "PT_H":
            return "H_type1"
        elif label[0] in ("NT_H", "NT_H_"):
            return "H_type3"
        elif label[0] in ("bT_H", "bT_H_"):
            return "H_type2"
        elif label[0] in ("BT_H", "BT_H_"):
            return "H_type1"
        else:
            return "not"
            # raise ValueError("not label")


class Stats(PKL):
    total_stats = {}
    counts = 0

    def __init__(self, pkl):
        super().__init__(pkl)
        self.merge2class()
        Stats.counts += 1

    def merge2class(self):
        for k, v in self.stats.items():
            Stats.total_stats[k] = Stats.total_stats.get(k, 0) + v


def main():
    print()


if __name__ == "__main__":
    main()
