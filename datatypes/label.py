from datatypes import Point

class Label(object):
    def __init__(self, row):
        self.pano_id = row[0]
        self.sv_image_x = float(row[1])
        self.sv_image_y = float(row[2])
        self.label_type = int(row[3])
        self.photographer_heading = float(row[4]) if row[4] is not None else None
        self.heading = float(row[5]) if row[5] is not None and len(row[5]) > 1 else None
        
    def to_row(self):
        row = []
        row.append(self.pano_id)
        row.append(self.sv_image_x)
        row.append(self.sv_image_y)
        row.append(self.label_type)
        row.append(self.photographer_heading)
        row.append(self.heading)
        return row
    
    def point(self):
        return Point(self.sv_image_x, self.sv_image_y)
    
    # def __str__(self):
    #     label = GSVutils.utils.label_from_int[self.label_type-1]
    #     return '{} at {}'.format(label, self.point() )
    
    @classmethod
    def header_row(cls):
        row = ['gsv_panorama_id', 'sv_image_x', 'sv_image_y', 'label_type_id', 'photographer_heading', 'heading', 'pitch', 'label_id']
        return row