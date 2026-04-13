(set! num-bands 10)
(set! resolution 64)
(set! k-points (list (vector3 0.0 0.0 0.0)))
(set! k-points (interpolate 4 k-points))
(define-param h 0.5)
(define-param r1 0.1)
(define-param r2 0.2)
(set! geometry-lattice (make lattice (size 1 1 4) (basis1  (vector3 1.0 0.0 0.0)) (basis2  (vector3 0.0 1.0 0.0)) ))
(set! geometry (list 
 (make block (size (vector3 1 1 h))
  (center (vector3 0.0 0.0 0.0))
  (material (make dielectric (epsilon 9.46))))
 (make cylinder (radius r1)
  (height h)
  (center (vector3 0.0 0.0 0.0))
  (material (make dielectric (epsilon 1))))
 (make cylinder (radius r2)
  (height h)
  (center (vector3 0.5 0.5 0.0))
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

(run-zeven )