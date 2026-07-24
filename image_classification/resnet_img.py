from util import Util

def main() :
    use_meta = False    # Indicate if use headers or not
    classifier = Util()

    # Load dataset
    classifier.load("/raid/mpsych/OMAMA/DATA/data/2d_resized_1024/", 
                    "/home/anya.tongprasith001/U54REC/release_to_header_mapping.pkl",
                    random_seed=10)
    
    # Sample data for test dataset
    classifier.load_test(use_meta)

    # Sample data for train/val dataset
    classifier.sample_data(use_meta)

    # Setup and train resnet50
    classifier.setupResnet(use_meta)
    classifier.train()

    # Make the model predict on test dataset
    classifier.predict()

if __name__ == "__main__":
    main()