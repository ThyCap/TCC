# Describe project

# TODO List:
OK -- "Tensorize loss" : generate loss tensor on domain
OK -- Automate loss heatmap generation
OK -- Plot heatmap with eventual holes 
OK (might have additions) -- Dashboard creation?
OK -- Add labels to axes
OK-- Add x and y vals to axes
OK -- Create Top and Bottom BCs
OK -- Plot N_u and N_f points on graph
OK -- Evolution plot on loss_bc and loss_pde
OK -- Statement on number of iterations
-- Understand why it stops
    OK -- 1. Total loss rises 
    -- ...
-- "Transposed" simple case
-- Study on Nu, Nf and Nf/Nu
-- Study on weights system

# Lessons learned: 
-- NN will optimize to a very sharp second derivative. Possible hypotheses:
    -- 1. overfitting due to number of layers/nodes
    -- 2. Sampling points fall out of sharp points
