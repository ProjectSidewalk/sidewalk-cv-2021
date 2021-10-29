WITH good_labels AS (
    SELECT label.label_id
    FROM sidewalk.label
    INNER JOIN sidewalk.label_validation ON label.label_id = label_validation.label_id
    INNER JOIN sidewalk.audit_task ON label.audit_task_id = audit_task.audit_task_id AND audit_task.street_edge_id != 27645
    WHERE label.deleted = false
      AND label.tutorial = FALSE
      AND label.street_edge_id != 27645
    GROUP BY label.label_id
    HAVING COUNT(CASE WHEN validation_result = 2  THEN 1 END) <= 2
        OR COUNT(CASE WHEN validation_result = 2 THEN 1 END) < (2 * COUNT(CASE WHEN  validation_result = 1 THEN 1 END))
)
SELECT label.gsv_panorama_id,
       label_point.sv_image_x,
       label_point.sv_image_y,
       label.label_type_id,
       label.photographer_heading,
       label_point.heading,
       label_point.pitch,
       label.label_id
FROM sidewalk.label
INNER JOIN sidewalk.label_point ON label.label_id = label_point.label_id
WHERE label.label_id IN (SELECT label_id FROM good_labels)
  AND label.label_type_id IN (1, 2, 3, 4)