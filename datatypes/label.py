from .point import Point

class Label(object):
    def __init__(self, row):
        self.pano_id = row['gsv_panorama_id']
        self.sv_image_x = float(row['sv_image_x'])
        self.sv_image_y = float(row['sv_image_y'])
        self.canvas_x = int(row['canvas_x'])
        self.canvas_y = int(row['canvas_y'])
        self.canvas_width = int(row['canvas_width'])
        self.canvas_height = int(row['canvas_height'])
        self.zoom = int(row['zoom'])
        self.label_type = int(row['label_type_id'])
        self.photographer_heading = float(row['photographer_heading']) if row['photographer_heading'] is not None else None
        self.photographer_pitch = float(row['photographer_pitch']) if row['photographer_pitch'] is not None else None
        self.heading = float(row['heading']) if row['heading'] is not None else None
        self.pitch = float(row['pitch']) if row['pitch'] is not None else None
        self.label_id = int(row['label_id']) if row['label_id'] is not None else None
        self.agree_count = int(row['agree_count']) if row['agree_count'] is not None else None
        self.disagree_count = int(row['disagree_count']) if row['disagree_count'] is not None else None
        self.notsure_count = int(row['notsure_count']) if row['notsure_count'] is not None else None
        self.deleted = row['deleted'] if row['deleted'] is not None else None
        self.tutorial = row['tutorial'] if row['tutorial'] is not None else None
        self.image_width = int(row['image_width']) if row['image_width'] is not None else None
        self.image_height = int(row['image_height']) if row['image_height'] is not None else None
        self.final_sv_image_x = None
        self.final_sv_image_y = None
        
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
        row.append(self.agree_count)
        row.append(self.disagree_count)
        row.append(self.notsure_count)
        row.append(self.deleted)
        row.append(self.tutorial)
        return row

    def finalize_sv_position(self, x, y):
        self.final_sv_image_x = x
        self.final_sv_image_y = y
    
    def point(self):
        if self.final_sv_image_x is None or self.final_sv_image_y is None:
            return None
        return Point(self.final_sv_image_x, self.final_sv_image_y)
    