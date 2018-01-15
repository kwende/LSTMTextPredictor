Once trained, run the following: 

python3 /usr/local/lib/python3.5/dist-packages/tensorflow/python/tools/freeze_graph.py \
--input_checkpoint=model.ckpt --input_graph=Profile.pb --output_node_names=add,Placeholder --output_graph=frozen.pb

This will get the node outputs needed for prediction and store them into frozen.pb. This is what gets deployed. 


