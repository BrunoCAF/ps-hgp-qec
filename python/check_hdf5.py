import h5py

with h5py.File("exploration.hdf5", 'r') as f:
    # f.visititems(lambda n, o: print(f"{n}: {o}"))
    for grp in f.values():
        print(grp.name)
        for a in grp.attrs:
            print(f"{grp}.{a} = {grp.attrs[a]}")

        for ds in grp.values():
            print(ds)