

def check_weights(model, modalities_no, seg_weights, loss_weights_g, loss_weights_d, tolerance=1e-10):
    assert sum(seg_weights) <= 1+tolerance and sum(seg_weights) >= 1-tolerance, 'seg weights should add up to 1'
    assert sum(loss_weights_g) <= 1+tolerance and sum(loss_weights_g) >= 1-tolerance, 'loss weights g should add up to 1'
    assert sum(loss_weights_d) <= 1+tolerance and sum(loss_weights_d) >= 1-tolerance, 'loss weights d should add up to 1'
    
    if model in ['DeepLIIF','DeepLIIFKD']:
        # +1 because input becomes an additional modality used in generating the final segmentation
        assert len(seg_weights) == modalities_no+1, 'seg weights should have the same number of elements as number of modalities to be generated'
        assert len(loss_weights_g) == modalities_no+1, 'loss weights g should have the same number of elements as number of modalities to be generated'
        assert len(loss_weights_d) == modalities_no+1, 'loss weights d should have the same number of elements as number of modalities to be generated'

    else:
        assert len(seg_weights) == modalities_no, 'seg weights should have the same number of elements as number of modalities to be generated'
        assert len(loss_weights_g) == modalities_no, 'loss weights g should have the same number of elements as number of modalities to be generated'
        assert len(loss_weights_d) == modalities_no, 'loss weights d should have the same number of elements as number of modalities to be generated'
