from .label import Label

class Panorama(object): 
    def __init__(self):
        self.feats = {}
        self.pano_id = None
        self.photog_heading = None
        self.photog_pitch = None
        self.width = None
        self.height = None

    def add_feature(self, row):
        feat = Label(row)
        if self.pano_id is None:
            self.pano_id = feat.pano_id
        assert self.pano_id == feat.pano_id
        
        if self.photog_heading is None:
            self.photog_heading = feat.photographer_heading

        if self.photog_pitch is None:
            self.photog_pitch = feat.photographer_pitch

        if feat.label_id not in self.feats:
            self.feats[feat.label_id] = feat

    def update_pano_size(self, width, height):
        self.width = width
        self.height = height
            
    def __hash__(self):
        return hash(self.pano_id)
    
    def all_feats(self):
        ''' iterate over all features, regardless of type '''
        for _, features in self.feats.items():
            yield features
    
    def __str__(self):
        s = 'pano{}\n'.format(self.pano_id)
        for feat in self.all_feats():
            s += '{}\n'.format(feat)
        return s
    
    def __len__(self):
        ''' return the total number of feats in this pano '''
        return len(self.feats)
