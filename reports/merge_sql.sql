SELECT *
    FROM sirh
    INNER JOIN evaluation USING (id_employee)
    INNER JOIN sond USING (id_employee);