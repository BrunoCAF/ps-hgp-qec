import h5py
import sys

# for grp in f.values():
#     print(grp.name)
#     for a in grp.attrs:
#         print(f"{grp}.{a} = {grp.attrs[a]}")

#     for ds in grp.values():
#         print(ds)

with h5py.File(sys.argv[1], 'r') as f:
    f.visititems(lambda n, o: print(f"{n}: {o}"))
