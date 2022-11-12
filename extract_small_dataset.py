from gmfpp.data_preparation import *

metadata = read_metadata("./data/metadata.csv")
metadata = drop_redundant_metadata_columns(metadata)

multi_cell_images_names = ["B02_s1_w1B1A7ADEA-8896-4C7D-8C63-663265374B72"]
metadata_small = filter_metadata_by_multi_cell_image_names(metadata, multi_cell_images_names)

save_metadata(metadata_small, "./data/metadata_small.csv")

print("completed!")
