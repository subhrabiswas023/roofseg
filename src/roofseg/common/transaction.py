from dataclasses import dataclass, field
from typing import Protocol, override


class StagedTransaction(Protocol):
    @property
    def is_failed(self) -> bool: ...
    def commit(self) -> None: ...
    def recover_if_failed(self) -> None: ...


class Transaction(Protocol):
    def stage(self) -> StagedTransaction: ...


       
@dataclass(frozen=True)  
class StagedBatchedTransaction(StagedTransaction):
    transactions: list[StagedTransaction] = field(default_factory=list)
    
    @override
    def commit(self) -> None:
        for tx in self.transactions:
            tx.commit()

    @property
    @override
    def is_failed(self) -> bool:
        return any(tx.is_failed for tx in self.transactions)

    @override
    def recover_if_failed(self) -> None:
        if not self.is_failed or not self.transactions:
            return

        first_transaction = self.transactions[0]

        if first_transaction.is_failed:
            for tx in self.transactions:
                tx.recover_if_failed()

        else:
            for tx in self.transactions:
                tx.commit()


@dataclass(frozen=True)
class BatchedTransaction(Transaction):
    transactions: list[Transaction] = field(default_factory=list)

    @override
    def stage(self) -> StagedBatchedTransaction:
        return StagedBatchedTransaction([tx.stage() for tx in self.transactions])