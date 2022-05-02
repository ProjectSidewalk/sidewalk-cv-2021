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
        self.heading = float(row[11]) if row[11] is not None else None
        self.pitch = float(row[12]) if row[12] is not None else None
        self.label_id = int(row[13]) if row[13] is not None else None
        self.agree_count = int(row[14]) if row[14] is not None else None
        self.disagree_count = int(row[15]) if row[15] is not None else None
        self.notsure_count = int(row[16]) if row[16] is not None else None
        self.deleted = row[17] == 't' if row[17] is not None else None
        self.tutorial = row[18] == 't' if row[18] is not None else None
        self.image_width = int(row[19]) if row[19] is not None else None
        self.image_height = int(row[20]) if row[20] is not None else None
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
    