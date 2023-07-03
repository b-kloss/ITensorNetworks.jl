alias(::SiteType"MyBoson") = SiteType"Qudit"()

"""
    space(::SiteType"Boson";
          dim = 2,
          conserve_qns = false,
          conserve_number = false,
          qnname_number = "Number")
Create the Hilbert space for a site of type "Boson".
Optionally specify the conserved symmetries and their quantum number labels.
"""
function ITensors.space(
    ::SiteType"MyBoson";
    dim=2,
    conserve_qns=false,
    conserve_number=false,
    qnname_number="Number",
  )
    if conserve_number && conserve_qns
      return [QN(qnname_number, n - 1) => 1 for n in 1:dim]
    elseif conserve_qns #dummy qns Optionally
      return [QN() => dim]
    end
    return dim
  end
  
ITensors.val(vn::ValName, st::SiteType"MyBoson") = val(vn, alias(st))

ITensors.state(sn::StateName, st::SiteType"MyBoson", s::Index) = state(sn, alias(st), s)

ITensors.op(on::OpName, st::SiteType"MyBoson", s::Index) = ITensors.op(on, alias(st), s)  ##Why do I need to put ITensors here?

##override fermionic (Jordan Wigner string) ordering operator F to identity for boson sites
ITensors.op(on::OpName"F", st::SiteType"MyBoson", s::Index) = ITensors.op(OpName("Id"), alias(st), s)
ITensors.op(on::OpName"F", st::SiteType"Qudit", s::Index) = ITensors.op(OpName("Id"), st, s)