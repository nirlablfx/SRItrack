from .otb import OTBDataset
from .uav import UAVDataset
from .lasot import LaSOTDataset
from .got10k import GOT10kDataset
from .dtb import DTBDataset
from .vot import VOTDataset
from .uav10fps import UAV10Dataset


class DatasetFactory(object):
    @staticmethod
    def create_dataset(**kwargs):
        """
        Args:
            name: dataset name 'OTB2015', 'LaSOT', 'UAV123', 'NFS240', 'NFS30',
                'VOT2018', 'VOT2016', 'VOT2018-LT'
            dataset_root: dataset root
            load_img: wether to load image
        Return:
            dataset
        """
        assert 'name' in kwargs, "should provide dataset name"
        name = kwargs['name']
        if 'OTB' in name:
            dataset = OTBDataset(**kwargs)
        elif 'LaSOT' == name:
            dataset = LaSOTDataset(**kwargs)
        elif 'UAV123_10fps' in name:
            dataset = UAV10Dataset(**kwargs)
        elif 'UAV' in name:
            dataset = UAVDataset(**kwargs)
        elif 'GOT10K' == name:
            dataset = GOT10kDataset(**kwargs)
        elif 'DTB70' == name:
            dataset = DTBDataset(**kwargs)
        elif 'VOT2016' == name:
            dataset = VOTDataset(**kwargs)
        elif 'VOT2018' == name:
            dataset = VOTDataset(**kwargs)
        elif 'VOT2019' == name:
            dataset = VOTDataset(**kwargs)

        else:
            raise Exception("unknow dataset {}".format(kwargs['name']))
        return dataset

