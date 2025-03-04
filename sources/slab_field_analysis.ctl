(set! num-bands 8)
(set! resolution 32)
(set! k-points (list (vector3 0.0 0.0 0.0) (vector3 0.05 0.02500000000000001 0.0)))
(set! k-points (interpolate 5 k-points))
(define-param r 0.35)
(set! geometry-lattice (make lattice (size 1.0 1.0 3.0) (basis1  (vector3 1.0 0.0 0.0)) (basis2  (vector3 0.5000000000000001 0.8660254037844388 0.0)) ))
(set! geometry (list 
 (make block (center (vector3 0.0 0.0 0.0))
  (size (vector3 1e+20 1e+20 0.4))
  (material (make dielectric (epsilon 20))))
 (make cylinder (center (vector3 0.0 0.0 0.0))
  (radius r)
  (height 0.4)
  (material (make dielectric (epsilon 1))))
))


(define (output-nonbloch-efield which-band)
    (get-efield which-band)
    (cvector-field-nonbloch! cur-field)
    (output-field-to-file -1 (string-append (get-filename-prefix)"e."))
)


(define (output-nonbloch-hfield which-band)
    (get-hfield which-band)
    (cvector-field-nonbloch! cur-field)
    (output-field-to-file -1 (string-append (get-filename-prefix)"h."))
)


(define (output-nonbloch-efield-x which-band)
    (get-efield which-band)
    (cvector-field-nonbloch! cur-field)
    (output-field-to-file 0 (string-append (get-filename-prefix)"e."))
)


(define (output-nonbloch-hfield-x which-band)
    (get-hfield which-band)
    (cvector-field-nonbloch! cur-field)
    (output-field-to-file 0 (string-append (get-filename-prefix)"h."))
)


(define (output-nonbloch-efield-y which-band)
    (get-efield which-band)   
    (cvector-field-nonbloch! cur-field)
    (output-field-to-file 1 (string-append (get-filename-prefix)"e."))
)


(define (output-nonbloch-hfield-y which-band)
    (get-hfield which-band)
    (cvector-field-nonbloch! cur-field)
    (output-field-to-file 1 (string-append (get-filename-prefix)"h."))
)


(define (output-nonbloch-efield-z which-band)
    (get-efield which-band)
    (cvector-field-nonbloch! cur-field)
    (output-field-to-file 2 (string-append (get-filename-prefix)"e."))
)


(define (output-nonbloch-hfield-z which-band)
    (get-hfield which-band)
    (cvector-field-nonbloch! cur-field)
    (output-field-to-file 2 (string-append (get-filename-prefix)"h."))
)

(run-zeven fix-efield-phase output-nonbloch-efield-z output-nonbloch-hfield-y)
(run-zodd fix-efield-phase output-nonbloch-efield-z output-nonbloch-hfield-y)