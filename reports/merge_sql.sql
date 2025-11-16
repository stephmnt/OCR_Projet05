SELECT *
    FROM public.sirh
    INNER JOIN public.evaluation USING (id_employee)
    INNER JOIN public.sond USING (id_employee);