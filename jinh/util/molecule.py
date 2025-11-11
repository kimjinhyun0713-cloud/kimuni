import numpy as np

molDic = {}
molDic["H3O"] = np.array([["H", -0.9628884076877837, 0, -0.2698998227870565, "h*"],
                            ["H", 0.48144420384389186, 0.8338858220671681, -0.2698998227870565, "h*"],
                            ["H", 0.48144420384389186, -0.8338858220671681, -0.2698998227870565, "h*"],
                            ["O", 0, 0, 0, "o*"]], dtype=object)
molDic["H2O"] = np.array([["H", 0.7906895737438433, 0.6122172800344493, 0, "h*"],
                            ["H", -0.7906895737438433, 0.6122172800344493, 0,  "h*"],
                            ["O", 0, 0, 0, "o*"]], dtype=object)
molDic["OH"] = np.array([["H", 1, 0, 0, "h*"],
                           ["O", 0, 0, 0, "o*"]], dtype=object)
molDic["CO3"] = np.array([["O", 0, 1.2, 0, "oc"],
                            ["O", 1.0392304845413263, -0.6, 0, "oc"],
                            ["O", -1.0392304845413263, -0.6, 0, "oc"],
                            ["C", 0, 0, 0, "co"]], dtype=object)
molDic["IO3"] =  np.array([['I', 0.0, 0.0, 0.0, "I"],
                             ['O', 1.41704, 1.03241, 0.5329200000000007, "O"],
                             ['O', 0.4380000000000006, -1.41788, 1.0711700000000004, "O"],
                             ['O', 0.8096500000000004, -0.77221, -1.4601600000000001, "O"]], dtype=object)



molWeight = {}
molWeight["H2O"] = 18.01528
molWeight["CO3"] = 60.009
molWeight["Ca"] = 40.0718
molWeight["I"] = 126.90447
molWeight["IO3"] = 174.90407

molCharge = {}
molCharge["O"] = -2
molCharge["H"] = 1
molCharge["Ca"] = 2
molCharge["Si"] = 4
molCharge["C"] = 4
molCharge["I"] = -1
molCharge["CO3"] = -2
molCharge["HCO3"] = -1
molCharge["H2CO3"] = 0
molCharge["H2O"] = 0
molCharge["CO2"] = 0
molCharge["OH"] = -1
molCharge["SiO4"] = -4

