from .point import Point

class Label(object):
    def __init__(self, row):
        self.pano_id = row[0]
        self.sv_image_x = float(row[1])
        self.sv_image_y = float(row[2])
        self.canvas_x = int(row[3])
        self.canvas_y = int(row[4])
        self.canvas_width = int(row[5])
        self.canvas_height = int(row[6])
        self.zoom = int(row[7])
        self.label_type = int(row[8])
        self.photographer_heading = float(row[9]) if row[9] is not None else None
        self.photographer_pitch = float(row[10]) if row[10] is not None else None
        self.heading = float(row[11]) if row[11] is not None and len(row[11]) > 1 else None
        self.pitch = float(row[12]) if row[12] is not None else None
        self.label_id = int(row[13]) if row[13] is not None else None
        
    def to_row(self):
        row = []
        row.append(self.pano_id)
        row.append(self.sv_image_x)
        row.append(self.sv_image_y)
        row.append(self.canvas_x)
        row.append(self.canvas_y)
        row.append(self.canvas_width)
        row.append(self.canvas_height)
        row.append(self.zoom)
        row.append(self.label_type)
        row.append(self.photographer_heading)
        row.append(self.photographer_pitch)
        row.append(self.heading)
        row.append(self.pitch)
        row.append(self.label_id)
        return row
    
    def point(self):
        return Point(self.sv_image_x, self.sv_image_y)
    
    # def __str__(self):
    #     label = GSVutils.utils.label_from_int[self.label_type-1]
    #     return '{} at {}'.format(label, self.point() )
    
    @classmethod
    def header_row(cls):
        row = ['gsv_panorama_id', 'sv_image_x', 'sv_image_y', 'canvas_x', 'canvas_y', 'canvas_width', 'canvas_height', 'zoom', 'label_type_id', 'photographer_heading', 'photographer_pitch', 'heading', 'pitch', 'label_id']
        return row