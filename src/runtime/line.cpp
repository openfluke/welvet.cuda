#include "welvet/runtime/line.hpp"

#include "welvet/runtime/training.hpp"
#include "welvet/systems/tween_alt.hpp"
#include "welvet/systems/tween_split.hpp"

#include <string>
#include <utility>

namespace welvet::runtime::line {
namespace {

TensorF32 clone_t(const TensorF32& t) {
  TensorF32 o;
  o.shape = t.shape;
  o.data = t.data;
  return o;
}

Status gap_credit(Credit credit) {
  return Status::Gap(std::string("runtime.line Credit ") +
                     std::to_string(static_cast<int>(credit)) + " (unsupported Line credit on CUDA)");
}

} // namespace

int in_flight(const State& st) { return static_cast<int>(st.flights.size()); }

Status tick(stack::Layer& ops, State* st, const TensorF32& input, const TensorF32* target,
            FlightRole role, double lr, TickResult* out, Credit credit) {
  if (!st || !out) {
    return Status::Err(StatusCode::InvalidArgument, "line: nil state/out");
  }
  if (input.data.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "line: empty input");
  }
  if (role == FlightRole::Train && (!target || target->data.empty())) {
    return Status::Err(StatusCode::InvalidArgument, "line: RoleTrain needs target");
  }

  if (static_cast<int>(credit) > static_cast<int>(Credit::FamilyTweenSplitSparse))
    return gap_credit(credit);
  if (credit != Credit::FamilyBP && credit != Credit::FamilyTweenSplit &&
      credit != Credit::FamilyTweenAlt)
    return gap_credit(credit);

  std::vector<stack::Child*> flat;
  stack::flatten_ops(ops, &flat);
  const int n = static_cast<int>(flat.size());
  if (n == 0) {
    return Status::Err(StatusCode::InvalidArgument, "line: empty step line");
  }

  Flight fl_new;
  fl_new.role = role;
  fl_new.ins.resize(static_cast<std::size_t>(n));
  fl_new.pres.resize(static_cast<std::size_t>(n));
  fl_new.posts.resize(static_cast<std::size_t>(n));
  if (role == FlightRole::Train) {
    fl_new.target = clone_t(*target);
  }
  st->flights.push_back(std::move(fl_new));

  out->loss = 0.0;
  out->popped_train = false;
  out->action_out = TensorF32{};

  std::vector<Flight> alive;
  alive.reserve(st->flights.size());
  for (std::size_t i = 0; i < st->flights.size(); ++i) {
    Flight& fl = st->flights[i];
    const int layer = fl.stage;
    if (layer < 0 || layer >= n) {
      continue;
    }
    const TensorF32* in = nullptr;
    if (layer == 0) {
      in = &input;
    } else {
      in = &fl.posts[static_cast<std::size_t>(layer - 1)];
    }
    if (!in || in->data.empty()) {
      return Status::Err(StatusCode::InvalidArgument,
                         std::string("line: step bubble at layer ") + std::to_string(layer));
    }

    TensorF32 pre, post;
    if (auto st_f = stack::child_forward(*flat[static_cast<std::size_t>(layer)], *in, &pre, &post);
        !st_f) {
      return Status::Err(st_f.code, std::string("line fwd layer ") + std::to_string(layer) + ": " +
                                        st_f.message);
    }
    fl.ins[static_cast<std::size_t>(layer)] = clone_t(*in);
    fl.pres[static_cast<std::size_t>(layer)] = std::move(pre);
    fl.posts[static_cast<std::size_t>(layer)] = std::move(post);
    fl.stage++;

    if (fl.stage < n) {
      alive.push_back(std::move(fl));
      continue;
    }

    if (fl.role == FlightRole::Action) {
      out->action_out = clone_t(fl.posts[static_cast<std::size_t>(n - 1)]);
      continue;
    }

    double loss = 0.0;
    TensorF32 gy;
    if (auto st_m = training::mse(fl.posts[static_cast<std::size_t>(n - 1)], fl.target, &loss);
        !st_m) {
      return st_m;
    }
    if (auto st_g = training::mse_grad(fl.posts[static_cast<std::size_t>(n - 1)], fl.target, &gy);
        !st_g) {
      return st_g;
    }
    out->loss = loss;
    out->popped_train = true;

    TensorF32 g = std::move(gy);
    if (credit == Credit::FamilyTweenSplit || credit == Credit::FamilyTweenAlt) {
      // Split: 1/N @ full lr. Alt: Split→Tween soft lr (frozen popped gap).
      // gin = copy gy if same len else zeros (no chain rule).
      for (int j = n - 1; j >= 0; --j) {
        Status st_t;
        if (credit == Credit::FamilyTweenAlt) {
          st_t = systems::tween_alt::apply_alt_child(
              *flat[static_cast<std::size_t>(j)], g, fl.ins[static_cast<std::size_t>(j)], lr, 1);
        } else {
          st_t = systems::tween_split::apply_split_child(
              *flat[static_cast<std::size_t>(j)], g, fl.ins[static_cast<std::size_t>(j)], lr);
        }
        if (!st_t) {
          return Status::Err(st_t.code,
                             std::string(credit == Credit::FamilyTweenAlt ? "line alt layer "
                                                                         : "line split layer ") +
                                 std::to_string(j) + ": " + st_t.message);
        }
        TensorF32 next;
        next.shape = fl.ins[static_cast<std::size_t>(j)].shape;
        next.data.assign(fl.ins[static_cast<std::size_t>(j)].data.size(), 0.0f);
        if (g.data.size() == next.data.size()) {
          next.data = g.data;
        }
        g = std::move(next);
      }
    } else {
      // FamilyBP (StepBP): reverse child_family_bp @ full lr with true gin.
      for (int j = n - 1; j >= 0; --j) {
        TensorF32 next;
        if (auto credit_status =
                stack::child_family_bp(*flat[static_cast<std::size_t>(j)], g,
                                       fl.ins[static_cast<std::size_t>(j)],
                                       fl.pres[static_cast<std::size_t>(j)], lr, &next);
            !credit_status) {
          return credit_status;
        }
        g = std::move(next);
      }
    }
  }
  st->flights = std::move(alive);
  return Status::Ok();
}

Status train_mse(stack::Layer& ops, State* st, const TensorF32& input, const TensorF32& target,
                 double lr, double* loss, Credit credit) {
  if (!loss) {
    return Status::Err(StatusCode::InvalidArgument, "line: nil loss");
  }
  TickResult tr;
  if (auto s = tick(ops, st, input, &target, FlightRole::Train, lr, &tr, credit); !s) return s;
  *loss = tr.loss;
  return Status::Ok();
}

} // namespace welvet::runtime::line
