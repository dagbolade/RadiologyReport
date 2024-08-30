import tensorflow as tf

# Load your model (use the correct format that is currently saved)
model = tf.keras.models.load_model('Radiography/Attention_with_cheXNet_full_model1')

# Save it in the HDF5 format
model.save('Radiography/Attention_with_cheXNet_full_model1.h5')
print("Model saved in HDF5 format successfully!")