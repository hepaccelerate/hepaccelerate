from hepaccelerate.utils import Dataset

# Define which columns we want to access
datastructures = {
    "Muon": [("Muon_Px", "float32"), ("Muon_Py", "float32"),],
    "Jet": [("Jet_E", "float32"), ("Jet_btag", "float32"),],
    "EventVariables": [
        ("NPrimaryVertices", "int32"),
        ("triggerIsoMu24", "bool"),
        ("EventWeight", "float32"),
    ],
}

# Define the dataset across the files
dataset = Dataset(
    "HZZ", ["data/HZZ.root"], datastructures, treename="events", datapath=""
)

# Load the data to memory
dataset.load_root()

# Jets in the first file
ifile = 0
jets = dataset.structs["Jet"][ifile]

# common offset array for jets
jets_offsets = jets.offsets
print(jets_offsets)

# data arrays
jets_energy = jets.E
jets_btag = jets.btag
print(jets_energy)
print(jets_btag)

ev_weight = dataset.eventvars[ifile]["EventWeight"]
print(ev_weight)
