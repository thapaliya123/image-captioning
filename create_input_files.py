from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='flickr8k',
                       karpathy_json_path='/content/gdrive/MyDrive/Colab Notebooks/EKbana/image_captioning_attention/image_caption_karpathy_json/dataset_flickr8k.json',
                       image_folder='/content/gdrive/MyDrive/Colab Notebooks/EKbana/image_captioning/flickr8k/images',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='/content/gdrive/MyDrive/Colab Notebooks/EKbana/image_captioning_attention/caption_data',
                       max_len=50)
