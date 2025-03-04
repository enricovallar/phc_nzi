(set! num-bands 8)
(set! resolution 64)
(set! k-points (list (vector3 0.0 0.0 0.0) (vector3 0.5 0.25000000000000006 0.0)))
(set! k-points (interpolate 20 k-points))
(define-param r1 0.3)
(define-param r2 0.2)
(set! geometry-lattice (make lattice (size 1 1 no-size) (basis1  (vector3 1.0 0.0 0.0)) (basis2  (vector3 0.5000000000000001 0.8660254037844388 0.0)) ))
(set! geometry (list 
 (make block (center (vector3 0.0 0.0 0.0))
  (size (vector3 1e+20 1e+20 1e+20))
  (material (make dielectric (epsilon 9.61))))
 (make cylinder (center (vector3 0.3333333333333333 0.3333333333333333 0.0))
  (radius r1)
  (height 1e+20)
  (material (make dielectric (epsilon 1))))
 (make cylinder (center (vector3 0.6666666666666666 0.6666666666666666 0.0))
  (radius r2)
  (height 1e+20)
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

(run-te fix-efield-phase output-nonbloch-efield-z output-nonbloch-hfield-y)
(run-tm fix-efield-phase output-nonbloch-efield-z output-nonbloch-hfield-y)