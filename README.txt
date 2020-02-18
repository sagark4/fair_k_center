--Get the algorithms.py file by Kleindessner et al. from https://github.com/matthklein/fair_k_center_clustering/blob/master/algorithms.py
--Rename it to kleindessner_etal_algorithms.py
--Put it in the same directory where this code is extracted (call this "base" directory)
--Installation of Anaconda preferred
--Tested on Python 3.7.4+, NetworkX 2.4, Numpy 1.17.4, Matplotlib 3.1.1, Seaborn 0.9.0, Pandas 0.25.3, Keras 2.2.4, keras-applications 1.0.8, keras-base 2.2.4, keras-preprocessing 1.1.0

--For getting the plots in Figures 2 and 3 run the following command
python random_fair_k_center.py

--Set up massive and real datasets (Note that setting up CelebA will take 10+ hours)

1) Set up CelebA dataset
  Download CelebA dataset from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
  Download the file img_align_celeba.zip
  Extract the images into "celeba/images/" inside the base directory
  Run the python file celeb_a.py: python celeb_a.py
  Run the following command inside celeba directory: for file in img_align_celeba_features_{00..99}.dat; do cat $file; done > ../img_align_celeba_features.dat
  Download the attributes file list_attr_celeba.txt from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html and place it in the base directory
  

2) Set up Sushi dataset
  Download the Sushi dataset sushi3-2016.zip from http://www.kamishima.net/sushi/
  Extract it in a directory named sushi3-2016 (this is default name) in the base directory

3) Set up Adult dataset
  Download the dataset from https://archive.ics.uci.edu/ml/datasets/Adult
  Create a directory named uci-adult and download all the files from the above into it
  Run the function process_adult_dataset() from our file algorithms.py
  Rename adult.data file to adult_attr.data in the uci-adult directory

4) For 100 GB dataset
  Run the functions generate_random_euclidean_dataset() and generate_random_euclidean_dataset_groups() from our file algorithms.py

Run the file algorithms.py to get a table like Table 1 in the main paper:
python algorithms.py

Uncomment the code in the end in algorithms.py to run the algorithms on the 100 GB euclidean dataset
  
