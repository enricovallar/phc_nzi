(set! num-bands 5)
(set! resolution 65)
(set! k-points (list (vector3 0.0 0.0 0.0)))
(set! k-points (interpolate 4 k-points))
(define-param h 1e+20)
(define-param r1 0.1)
(define-param r2 0.2)
(set! geometry-lattice (make lattice (size 1 1 no-size) (basis1  (vector3 1.0 0.0 0.0)) (basis2  (vector3 0.5000000000000001 0.8660254037844388 0.0)) ))
(set! geometry (list 
 (make block (size (vector3 1 1 h))
  (center (vector3 0.0 0.0 0.0))
  (material (make dielectric (epsilon 1))))
 (make cylinder (radius r1)
  (height h)
  (center (vector3 0.0 0.0 0.0))
  (material (make dielectric (epsilon 12.11))))
 (make cylinder (radius r2)
  (height h)
  (center (vector3 0.3333333333333333 0.3333333333333333 0.0))
  (material (make dielectric (epsilon 12.11))))
 (make cylinder (radius r2)
  (height h)
  (center (vector3 0.6666666666666666 0.6666666666666666 0.0))
  (material (make dielectric (epsilon 12.11))))
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


; --- C4v Definitions for Square Lattice: basis1=(1,0), basis2=(0,1) ---
(define C4-s (matrix3x3 (vector3 0 1 0) (vector3 -1 0 0) (vector3 0 0 1)))
(define C2-s (matrix3x3 (vector3 -1 0 0) (vector3 0 -1 0) (vector3 0 0 1)))
(define sv-s (matrix3x3 (vector3 1 0 0) (vector3 0 -1 0) (vector3 0 0 1)))  
(define sd-s (matrix3x3 (vector3 0 1 0) (vector3 1 0 0) (vector3 0 0 1)))  

(define (display-symmetries-c4v)
  (if (vector3= current-k (vector3 0 0 0))
      (begin
        (print "SYM_DATA_START_" parity "
")
        (map (lambda (b)
               (print parity "," b 
                      ",C4=" (compute-symmetry b C4-s (vector3 0 0 0)) 
                      ",C2=" (compute-symmetry b C2-s (vector3 0 0 0)) 
                      ",sv=" (compute-symmetry b sv-s (vector3 0 0 0)) 
                      ",sd=" (compute-symmetry b sd-s (vector3 0 0 0)) "
"))
             (arith-sequence 1 1 num-bands))
        (print "SYM_DATA_END_" parity "
"))))


; --- C6v Definitions for Hexagonal Lattice: basis1=(1,0), basis2=(0.5, 0.866) ---
(define C6-h (matrix3x3 (vector3 0 1 0) (vector3 -1 1 0) (vector3 0 0 1))) 
(define C3-h (matrix3x3 (vector3 -1 1 0) (vector3 -1 0 0) (vector3 0 0 1)))
(define C2-h (matrix3x3 (vector3 -1 0 0) (vector3 0 -1 0) (vector3 0 0 1))) 
(define sv-h (matrix3x3 (vector3 1 0 0) (vector3 1 -1 0) (vector3 0 0 1)))  
(define sd-h (matrix3x3 (vector3 0 1 0) (vector3 1 0 0) (vector3 0 0 1)))   

(define (display-symmetries-c6v)
  (if (vector3= current-k (vector3 0 0 0))
      (begin
        (print "SYM_DATA_START_" parity "
") ; Unique start tag
        (map (lambda (b)
               (print parity "," b 
                      ",C6=" (compute-symmetry b C6-h (vector3 0 0 0)) 
                      ",C3=" (compute-symmetry b C3-h (vector3 0 0 0)) 
                      ",C2=" (compute-symmetry b C2-h (vector3 0 0 0)) 
                      ",sv=" (compute-symmetry b sv-h (vector3 0 0 0)) 
                      ",sd=" (compute-symmetry b sd-h (vector3 0 0 0)) "
"))
             (arith-sequence 1 1 num-bands))
        (print "SYM_DATA_END_" parity "
")))) ; Unique end tag

(run-tm )_345