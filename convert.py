import tensorflow as tf

# Load your existing model
model = tf.keras.models.load_model('Radiography/Attention_with_cheXNet_full_model')

# Save in .keras format
model.save('Radiography/Attention_with_cheXNet_full_model.keras')

# Or save in .h5 format
model.save('Radiography/Attention_with_cheXNet_full_model.h5')